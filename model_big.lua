require 'cunn'
require 'cudnn'
require 'nngraph'
require 'nnx'
require 'Reparametrize'
require 'model'

function getEncoder(nFeatures, sSize, zSize, zSample, sSample,
		    nChannels, hInput, wInput)
   -- Returns an encoder. It expects inputs of size nChannels*hInput*wInput,
   -- and produces a code z of size (2,nFeatures*hInput*wInput/8), where
   -- the first coordinate is the mean and the second is the variance,
   -- and a code s of size sSize.
   local input = nn.Identity()()
   local x = input
   assert((hInput <= 32) and (wInput <= 32), 'input size must be <= 32x32')
   if (hInput < 32) or (wInput < 32) then
      local padh, padw = 32 - hInput, 32 - wInput
      x = nn.SpatialZeroPadding(math.floor(padw/2), math.ceil(padw/2),
				math.floor(padh/2), math.ceil(padh/2))(x)
      hInput, wInput = 32, 32
   end
   local x = convblock(nChannels  , nFeatures  , 3, 1)(x) -- /1
   local x = convblock(nFeatures  , nFeatures  , 3, 1)(x) -- /1
   local x = convblock(nFeatures  , nFeatures*2, 2, 2)(x) -- /2
   local x = convblock(nFeatures*2, nFeatures*2, 3, 1)(x) -- /2
   local x = convblock(nFeatures*2, nFeatures*4, 2, 2)(x) -- /4
   local x = convblock(nFeatures*4, nFeatures*4, 3, 1)(x) -- /4
   local x = convblock(nFeatures*4, nFeatures*8, 2, 2)(x) -- /8
   local x = convblock(nFeatures*8, nFeatures*8, 3, 1)(x) -- /8
   local npix = nFeatures*8*hInput*wInput/8/8
   local x = nn.View(npix):setNumInputDims(3)(x)
   local s, z
   if zSample then
      z = nn.Linear(npix, zSize*2):cuda()(x)
      z = nn.View(2, zSize):setNumInputDims(1)(z)
   else
      z = nn.Linear(npix, zSize):cuda()(x)
   end
   if sSample then
      s = nn.Linear(npix, sSize*2):cuda()(x)
      s = nn.View(2, sSize):setNumInputDims(1)(s)
   else
      s = nn.Linear(npix, sSize):cuda()(x)
   end
   return nn.gModule({input}, {z, s}):cuda()
end

function getDecoder(nFeatures, zSize, sSize, zSample, sSample,
		    nChannels, hInput, wInput)
   local inputz, inputs = nn.Identity()(), nn.Identity()()
   local z, s = inputz, inputs
   if zSample then
      z = nn.SplitTable(2)(z)
      z = nn.Reparametrize(zSize)(z)
   end
   if sSample then
      s = nn.SplitTable(2)(s)
      s = nn.Reparametrize(sSize)(s)
   end
   local npix = nFeatures*8*4*4
   z = nn.Linear(zSize, npix):cuda()(z)
   s = nn.Linear(sSize, npix):cuda()(s)
   local x = nn.CAddTable(){z, s}
   local x = nn.View(nFeatures*8, 4, 4)(x)
   local x = convblock(nFeatures*8, nFeatures*8, 3, 1)(x)
   local x = convblock(nFeatures*8, nFeatures*4, 2, 1/2)(x)
   local x = convblock(nFeatures*4, nFeatures*4, 3, 1)(x)
   local x = convblock(nFeatures*4, nFeatures*2, 2, 1/2)(x)
   local x = convblock(nFeatures*2, nFeatures*2, 3, 1)(x)
   local x = convblock(nFeatures*2, nFeatures  , 2, 1/2)(x)
   local x = convblock(nFeatures  , nFeatures  , 3, 1)(x)
   local x = conv     (nFeatures  , nChannels  , 3, 1)(x)
   assert((hInput <= 32) and (wInput <= 32), 'input size must be <= 32x32')
   if (hInput < 32) or (wInput < 32) then
      local padh, padw = 32 - hInput, 32 - wInput
      x = nn.SpatialZeroPadding(-math.floor(padw/2), -math.ceil(padw/2),
				-math.floor(padh/2), -math.ceil(padh/2))(x)
   end
   local x = nn.Tanh()(x)
   return nn.gModule({inputz, inputs}, {x}):cuda()
end

function getDiscriminator(nFeatures, nSid, nChannels, hInput, wInput)
   if type(nSid) == 'table' then
      assert(nFeatures*hInput*wInput/8/8 % #nSid == 0,
	     "nFeatures must be a multiple of " .. #nSid .. " (multi-class)")
   end

   local inputpix, inputsid = nn.Identity()(), nn.Identity()()
   local x = inputpix
   local sid = nn.FunctionWrapper(
      function(self) self.gradInput = torch.CudaTensor() end,
      function(self, input) return input end,
      function(self, input, gradOutput)
	 self.gradInput:resizeAs(input):zero()
	 return self.gradInput
      end)(inputsid) --because lookuptable produce no gradient
   assert((hInput <= 32) and (wInput <= 32), 'input size must be <= 32x32')
   if (hInput < 32) or (wInput < 32) then
      local padh, padw = 32 - hInput, 32 - wInput
      x = nn.SpatialZeroPadding(math.floor(padw/2), math.ceil(padw/2),
				math.floor(padh/2), math.ceil(padh/2))(x)
      hInput, wInput = 32, 32
   end

   local function getLookup(n, upsampling)
      local out = nn.Sequential()
      if type(nSid) == 'table' then
	 out:add(getMultiLookupTable(nSid, nFeatures*n*hInput*wInput/8/8/#nSid))
      else
	 out:add(nn.LookupTable(nSid, nFeatures*n*hInput*wInput/8/8))
      end
      out:add(nn.View(nFeatures*n, hInput/8, wInput/8):setNumInputDims(1))
      if upsampling ~= 1 then
	 out:add(nn.SpatialUpSamplingNearest(upsampling))
      end
      return out
   end

   local x = conv(nChannels, nFeatures, 3, 1)(x)
   local x = nn.LeakyReLU(0.2)(x)
   local x = conv(nFeatures, nFeatures, 2, 2)(x)
   local lut1 = getLookup(1, 4)(sid)
   local x = nn.CAddTable(){x, lut1}
   local x = nn.SpatialBatchNormalization(nFeatures):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local x = conv(nFeatures, nFeatures*2, 2, 2)(x)
   local x = nn.SpatialBatchNormalization(nFeatures*2):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local x = conv(nFeatures*2, nFeatures*2, 3, 1)(x)
   local x = nn.SpatialBatchNormalization(nFeatures*2):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local lut2 = getLookup(2, 2)(sid)
   local x = nn.CAddTable(){x, lut2}
   local x = nn.SpatialBatchNormalization(nFeatures*2):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local x = conv(nFeatures*2, nFeatures*4, 2, 2)(x)
   local x = nn.SpatialBatchNormalization(nFeatures*4):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local x = conv(nFeatures*4, nFeatures*4, 3, 1)(x)
   local lut3 = getLookup(4, 1)(sid)
   local x = nn.CAddTable(){x, lut3}
   local x = nn.SpatialBatchNormalization(nFeatures*4):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local x = conv(nFeatures*4, nFeatures*4, 3, 1)(x)
   local x = nn.SpatialBatchNormalization(nFeatures*4):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local npix = nFeatures*4*hInput*wInput/8/8
   local x = nn.View(npix):setNumInputDims(3)(x)
   local x = nn.Dropout()(x)
   local x = nn.Linear(npix, 1):cuda()(x)
   local x = nn.Sigmoid()(x)
   return nn.gModule({inputpix, inputsid}, {x}):cuda()
end

function getClassifierFromS(sSize, nClasses, sSample, nHidden, dropout)
   local classifier = nn.Sequential()
   if sSample then
      classifier:add(nn.SplitTable(2))
      classifier:add(nn.Reparametrize(sSize))
   end
   classifier:add(nn.Linear(sSize, nHidden):cuda())
   --classifier:add(nn.BatchNormalization(nHidden, nil, nil, false))
   classifier:add(nn.ReLU())
   classifier:add(nn.Dropout())
   if type(nClasses) == 'table' then
      local ct = nn.ConcatTable()
      classifier:add(ct)
      for i = 1, #nClasses do
	 local seq = nn.Sequential()
	 ct:add(seq)
	 seq:add(nn.Linear(nHidden, nClasses[i]):cuda())
	 seq:add(nn.LogSoftMax())
      end
   else
      classifier:add(nn.Linear(nHidden, nClasses):cuda())
      classifier:add(nn.LogSoftMax())
   end
   return classifier:cuda()
end
