require 'cutorch'
require 'optim'
require 'dataset'
require 'util'
require 'cunn'
require 'nngraph'
require 'KLDCriterion'
require 'adversarial_criterion'

----------------------------------- SETTINGS -----------------------------------

local cmd = torch.CmdLine()
-- device settings
cmd:option('--devid', 1, 'gpu id')
cmd:option('--saveName', '/scratch/michael/content_model_new.t7', 'Save name')
cmd:option('--displayPort', 8042, 'Post used for the module display. 0 to disable')

-- data
cmd:option('--dataset', 'mnist', 'Dataset [mnist|sprites|norb|celebA|yaleB]')
cmd:option('--datapath', 'data', 'Root directory for the datasets')

-- training hyperparameters
cmd:option('--nEpoches', 100000, 'Number of "epoches"')
cmd:option('--nIters', 500, 'Number of minibatches per "epoch"')
cmd:option('--batchSize', 16, 'Batch size')
cmd:option('--learningRateGen', 0.01, 'Learning rate (generative model)')
cmd:option('--learningRateAdv', 0.01, 'Learning rate (discrminator)')
cmd:option('--beta1', 0.5, 'beta1 parameter of adam (ignored for sgd)')
cmd:option('--optimGen', 'adam', '[sgd|adam]')
cmd:option('--optimDisc', 'sgd', '[sgd|adam]')
cmd:option('--labelReal', 0.95, "'Real' target")
cmd:option('--labelFake', 0.0, "'Fake' target")

-- relative weights
cmd:option('--recweight', 0.5, 'Weighting of the reconstruction')
cmd:option('--swap1weight', 1, 'Weighting of the swapping (same s)')
cmd:option('--swap2weight', 0.01, 'Weighting of the swapping (different s)')
cmd:option('--classweight', 0, 'Weighting of the classifier')
cmd:option('--klweightZ', 0.1, 'Weighting of the KL divergence term (S)')
cmd:option('--klweightS', 0, 'Weighting of the KL divergence term (Z)')
cmd:option('--samplingweight', 0, 'Weighting of the generation with sampled z')

-- model
cmd:option('--modelSize', 'big', 'Model architecture (depth) [big|small]')
cmd:option('--kSize', 4, 'Convolution kernel size')
cmd:option('--sSize', 64, 'Size of the s vector')
cmd:option('--zSize', 512, 'Size of the z vector')
cmd:option('--nFeatures', 16, 'Number of feature planes at the first layer')
cmd:option('--classifierNHiddenUnits', 256, 'Classifier number of hidden units')
opt = cmd:parse(arg)

nngraph.setDebug(false)
torch.setnumthreads(2)
torch.manualSeed(1)
cutorch.setDevice(opt.devid)
if opt.displayPort ~= 0 then
   display = require 'display'
   display.configure{server='vine12.cs.nyu.edu', port=opt.displayPort}
end

------------------------------------- MODEL ------------------------------------

if opt.dataset ~= 'mnist' then
   opt.h, opt.w = 32, 32
end
local getBatch = get_getBatch(opt)

if opt.modelSize == 'big' then
   require 'model_big'
else
   assert(opt.modelSize == 'small')
   require 'model_small'
end
local encoder = getEncoder(opt.nFeatures, opt.sSize, opt.zSize,
			   opt.klweightZ ~= 0, opt.klweightS ~= 0,
			   opt.nChannels, opt.h, opt.w)

local decoder = getDecoder(opt.nFeatures, opt.zSize, opt.sSize,
			   opt.klweightZ ~= 0, opt.klweightS ~= 0,
			   opt.nChannels, opt.h, opt.w)
local discriminator = getDiscriminator(opt.nFeatures, opt.nClasses,
				       opt.nChannels, opt.h, opt.w)
local classifier = getClassifierFromS(opt.sSize, opt.nClasses,
				      opt.klweightS ~= 0,
				      opt.classifierNHiddenUnits, false)

local lossAdv = nn.AdvCriterion(discriminator, opt.labelReal, opt.labelFake)
local model, loss = nil, nn.ParallelCriterion()
do -- this is where we connect all the parts
   loss:add(nn.MSECriterion(), opt.recweight)
   loss:add(nn.MSECriterion(), opt.swap1weight)
   loss:add(lossAdv, opt.swap2weight)
   local x, sid = nn.Identity()(), nn.Identity()()
   local z, s = encoder(x):split(2)
   local z12 = nn.Narrow(1, 1, 2*opt.batchSize)(z)
   local z2 = nn.Narrow(1, opt.batchSize+1, opt.batchSize)(z)
   local z3 = nn.Narrow(1, opt.batchSize*2+1, opt.batchSize)(z)
   local z32 = nn.JoinTable(1){z3, z2}
   local s1 = nn.Narrow(1, 1, opt.batchSize)(s)
   local s2 = nn.Narrow(1, opt.batchSize+1, opt.batchSize)(s)
   local s21 = nn.JoinTable(1){s2, s1}
   local s23 = nn.Narrow(1, opt.batchSize+1, 2*opt.batchSize)(s)
   local z_123_12_32 = nn.JoinTable(1){z, z12, z32}
   local s_123_21_23 = nn.JoinTable(1){s, s21, s23}
   local y_rec_swap1_swap2 = decoder{z_123_12_32, s_123_21_23}
   local y_rec = nn.Narrow(1,1, opt.batchSize*3)(y_rec_swap1_swap2)
   local y_swap1 = nn.Narrow(1, opt.batchSize*3+1, opt.batchSize*2)(y_rec_swap1_swap2)
   local y_swap2 = nn.Narrow(1, opt.batchSize*5+1, opt.batchSize*2)(y_rec_swap1_swap2)
   local sid23 = nn.Narrow(1,opt.batchSize+1, 2*opt.batchSize)(sid)
   local y_adv = nn.Identity(){y_swap2, sid23}
   local output = {y_rec, y_swap1, y_adv}
   if opt.classweight ~= 0 then
      output[1+#output] = classifier(s)
      if type(opt.nClasses) == 'table' then
	 lossNLL = nn.ParallelCriterion():cuda()
	 for i = 1, #opt.nClasses do
	    lossNLL:add(nn.ClassNLLCriterion(), 1/#opt.nClasses)
	 end
	 loss:add(lossNLL, opt.classweight)
      else
	 lossNLL = nn.ClassNLLCriterion():cuda()
	 loss:add(lossNLL, opt.classweight) 
      end
   end
   if opt.klweightZ ~= 0 then
      output[1+#output] = nn.SplitTable(2)(z)
      loss:add(nn.KLDivCriterion(), opt.klweightZ*6/(opt.nChannels*opt.h*opt.w))
   end
   if opt.klweightS ~= 0 then
      output[1+#output] = nn.SplitTable(2)(s)
      loss:add(nn.KLDivCriterion(), opt.klweightS*6/(opt.nChannels*opt.h*opt.w))
   end
   if opt.samplingweight ~= 0 then
      --nngraph cannot have a note without input:
      require 'nnx'
      local z_sample = nn.FunctionWrapper(
	 function(self)
	    self.output = torch.CudaTensor()
	    self.gradInput = torch.CudaTensor()
	 end,
	 function(self, input)
	    self.output:resizeAs(input):normal()
	    return self.output
	 end,
	 function(self, input, gradOutput)
	    self.gradInput:resizeAs(input):zero()
	    return self.gradInput
	 end)(z32)
      local y_23_sample = decoder:clone(){z_sample, s23}
      output[1+#output] = nn.Identity(){y_23_sample, sid23}
      loss:add(lossAdv:clone(), opt.samplingweight)
   end
   model = nn.gModule({x, sid}, output):cuda()
end
loss:cuda()

local function makeSidTarget(sid)
   if type(opt.nClasses) == 'table' then
      local out = {}
      for i = 1, #opt.nClasses do
	 out[i] = sid:select(2,i)
      end
      return out
   else
      return sid
   end
end

----------------------------------- TRAINING -----------------------------------

local wGen, dwGen = getParametersOfNetworks(encoder, decoder, classifier)
local configGen = {learningRate = opt.learningRateGen, beta1 = opt.beta1}
local configAdv = {learningRate = opt.learningRateAdv, beta1 = opt.beta1}
local optimfunGen = optim[opt.optimGen]
local optimfunDisc = optim[opt.optimDisc]
local y_adv_gen = nil
for iEpoch = 1, opt.nEpoches do
   print("Starting epoch " .. iEpoch)
   local sum_class_err = 0
   lossAdv:resetErrors()

   -- train
   for iIter = 1, opt.nIters do
      local data = {getBatch(opt.batchSize, 'train')}
      local x, sid = flatBatch(data[1]), flatBatch(data[2])
      local x12 = x:narrow(1,1,2*opt.batchSize)
      local target = {x, x12, {}, makeSidTarget(sid), {}}

      -- train generator
      local function fevalGen(params)
	 assert(params == wGen)
	 dwGen:zero()
	 local output = model:forward{x, sid}
	 y_adv_gen = output[3]
	 local err = loss:forward(output, target)
	 local doutput = loss:backward(output, target)
	 model:backward({x, sid}, doutput)
	 if lossNLL ~= nil then
	    sum_class_err = sum_class_err + lossNLL.output
	 end
	 return err, dwGen
      end
      optimfunGen(fevalGen, wGen, configGen)

      -- train discriminator          
      if (opt.learningRateAdv ~= 0) and (opt.swap2weight ~= 0) then
	 local x23 = x:narrow(1,opt.batchSize+1, 2*opt.batchSize)
	 local sid23 = sid:narrow(1,opt.batchSize+1, 2*opt.batchSize)
	 local fevalAdv = lossAdv:getFeval({x23, sid23}, y_adv_gen)
	 optimfunDisc(fevalAdv, lossAdv.weight, configAdv)
      end
   end
   print("Adv error= " .. lossAdv:getError())
   print("Class error= " .. sum_class_err / opt.nIters)

   -- test
   local x, sid = getBatch(opt.batchSize, 'test')
   local xFlat, sidFlat = flatBatch(x), flatBatch(sid)
   local output = model:forward{xFlat, sidFlat}
   local yrec = unflatBatch(output[1], 3)
   local yswap1 = unflatBatch(output[2], 2)
   local yswap2 = unflatBatch(output[3][1], 2)
   win = displayTensors({x[1], x[2], x[3], yrec[1], yrec[2], yrec[3],
			 yswap1[1], yswap1[2], yswap2[1], yswap2[2]}, win)
   encoder:clearState()
   decoder:clearState()
   classifier:clearState()
   discriminator:clearState()
   collectgarbage()
   torch.save(opt.saveName, {opt=opt, encoder=encoder, decoder=decoder, classifier=classifier, discriminator=disciminator})
end
