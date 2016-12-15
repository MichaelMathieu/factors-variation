--[[
   This file provides the function 'get_getBatch'.
   It takes a table 'opt' and returns a function 'getBatch'.
--]]

require 'paths'

local function get_dataset_iterator(data, labels, dim)
   local i = 1
   return function()
      if i > data:size(dim) then
	 return nil
      else
	 i = i + 1
	 return data:select(dim, i-1), labels:select(dim, i-1)
      end
   end
end

local function sort_dataset(iterator, nClasses, sampleSize)
   local nSamplesPerClass = {}
   for i = 1, nClasses do
      nSamplesPerClass[i] = 0
   end
   for sample, label in iterator() do
      if type(label) ~= 'number' then
	 label = label[1]
	 assert(type(label) == 'number')
      end
      nSamplesPerClass[label] = nSamplesPerClass[label] + 1
   end
   local out = {}
   for i = 1, nClasses do
      out[i] = torch.Tensor(nSamplesPerClass[i], unpack(sampleSize))
   end
   for sample, label in iterator() do
      if type(label) ~= 'number' then
	 label = label[1]
      end
      out[label][nSamplesPerClass[label] ]:copy(sample)
      nSamplesPerClass[label] = nSamplesPerClass[label] - 1
   end
   for i = 1, nClasses do
      assert(nSamplesPerClass[i] == 0)
   end
   return out
end

local function recursive_reduce(f_map, f_reduce, x)
   if type(x) ~= 'table' then
      return f_map(x)
   else
      local out = nil
      for k,v in pairs(x) do
	 local red = recursive_reduce(f_map, f_reduce, v)
	 if out == nil then
	    out = red
	 else
	    out = f_reduce(out, red)
	 end
      end
      return out
   end
end

local function normalize_data(trainset, ...)
   --if type(trainset) == 'table' then
   local min = recursive_reduce(torch.min, math.min, trainset)
   local max = recursive_reduce(torch.max, math.max, trainset)
   for k,v in pairs{trainset, ...} do
      recursive_reduce(function(x) x:add(-min):div((max-min)/2):add(-1) end, function() end, v)
   end
end

local X_cpu, Sid_cpu = torch.Tensor(), torch.LongTensor()
local X_gpu, Sid_gpu = torch.CudaTensor(), torch.CudaTensor()

local function getMnist_getBatch(opt)
   datapath = opt.datapath .. '/mnist/'
   opt.nClasses = 10
   opt.nChannels = 1
   opt.h = opt.h or 28
   opt.w = opt.w or 28
   assert(opt.h == 28, 'not implemented')
   assert(opt.w == 28, 'not implemented')
   local sets = {
      train = torch.load(paths.concat(datapath, 'train_28x28.th7')),
      test = torch.load(paths.concat(datapath, 'test_28x28.th7'))
   }
   local data, labels, sorted = {}, {}, {}
   for k,v in pairs(sets) do
      data[k] = v.data:type(torch.getdefaulttensortype())
      labels[k] = v.labels
   end
   normalize_data(data.train, data.test)
   for k,v in pairs(data) do
      sorted[k] = sort_dataset(function() return get_dataset_iterator(v, labels[k], 1) end,
			       opt.nClasses, {opt.nChannels, opt.h, opt.w})
   end
   
   return function(batchSize, set)
      batchSize = batchSize or opt.batchSize
      local sorted_dataset_set = sorted[set]
      assert(sorted_dataset_set ~= nil, 'Unknown set ' .. set)
      X_cpu:resize(3, batchSize, opt.nChannels, opt.h, opt.w)
      X_gpu:resize(3, batchSize, opt.nChannels, opt.h, opt.w)
      Sid_cpu:resize(3, batchSize):random(1, opt.nClasses)
      Sid_cpu[2]:copy(Sid_cpu[1])
      Sid_gpu:resize(3, batchSize)
      for i = 1, 3 do
	 for j = 1, batchSize do
	    local data = sorted_dataset_set[Sid_cpu[i][j] ]
	    X_cpu[i][j]:copy(data[torch.random(data:size(1))])
	 end
      end
      X_gpu:copy(X_cpu)
      Sid_gpu:copy(Sid_cpu)
      return X_gpu, Sid_gpu
   end
end

local function getSprites_getBatch(opt)
   local datapath = opt.datapath .. '/sprites/'
   opt.nChannels = 3
   opt.h = opt.h or 64
   opt.w = opt.w or 64
   local sets = {
      train = torch.load(paths.concat(datapath, 'train.t7b')),
      test = torch.load(paths.concat(datapath, 'test.t7b')),
      val = torch.load(paths.concat(datapath, 'val.t7b'))
   }
   local data, labels = {}, {}
   opt.nClasses = {0, 0, 0, 0, 0, 0, 0}
   for k,v in pairs(sets) do
      data[k] = v.data
      labels[k] = v.labels
      for i = 1, #data[k] do
	 local label = labels[k][i]
	 for j = 1, label:size(1) do
	    opt.nClasses[j] = math.max(opt.nClasses[j], label[j]+1)
	 end
	 for j = 1, #data[k][i] do
	    data[k][i][j] = data[k][i][j]:type(torch.getdefaulttensortype())
	 end
      end
   end
   normalize_data(data.train, data.test, data.val)
   local idx = torch.Tensor()

   return function(batchSize, set)
      batchSize = batchSize or opt.batchSize
      assert(data[set] ~= nil, 'Unknown set ' .. set)
      X_cpu:resize(3, batchSize, opt.nChannels, opt.h, opt.w)
      X_gpu:resize(3, batchSize, opt.nChannels, opt.h, opt.w)
      idx:resize(3, batchSize):random(1, #labels[set])
      idx[2]:copy(idx[1])
      Sid_cpu:resize(3, batchSize, 7)
      Sid_gpu:resize(3, batchSize, 7)
      X_cpu:fill(-1)
      for i = 1, 3 do
	 for j = 1, batchSize do
	    local sprite = data[set][idx[i][j] ]
	    local style = labels[set][idx[i][j] ]
	    local numClip = torch.random(#sprite)
	    local idxInClip = torch.random(sprite[numClip]:size(1))
	    image.scale(X_cpu[i][j], sprite[numClip][idxInClip])
	    Sid_cpu[i][j]:copy(style):add(1)
	 end
      end
      X_gpu:copy(X_cpu)
      Sid_gpu:copy(Sid_cpu)
      return X_gpu, Sid_gpu
   end
end

local function getYaleB_getBatch(opt)
   local datapath = opt.datapath .. '/yaleB/'
   opt.nChannels = 1
   opt.h = opt.h or 64
   opt.w = opt.w or 64
   local sets = {}
   if opt.dataset == 'yaleB' then
      sets.train = torch.load(paths.concat(datapath, 'train.t7b'))
      sets.test = torch.load(paths.concat(datapath, 'test.t7b'))
   elseif opt.dataset == 'yaleBExtended' then
      sets.train = torch.load(paths.concat(datapath, 'train_aug.t7b'))
      sets.test = torch.load(paths.concat(datapath, 'test_aug.t7b'))
   else
      error("What is this dataset: " .. opt.dataset .. " ?")
   end
   opt.nClasses = #sets.train
   local data = {}
   for k,v in pairs(sets) do
      data[k] = v
      for i = 1, #v do
	 data[k][i] = v[i].data:type(torch.getdefaulttensortype())
      end
   end
   normalize_data(data.train, data.test)

   local downscaler = nn.Identity():cuda()
   if opt.h ~= 64 then
      assert((opt.h == opt.w) and (64 % opt.h == 0), 'not implemented')
      require 'cunn'
      local p = 64/opt.h
      downscaler = nn.SpatialAveragePooling(p, p, p, p):cuda()
   end
   
   return function(batchSize, set)
      batchSize = batchSize or opt.batchSize
      assert(data[set] ~= nil, 'Unknown set ' .. set)
      X_cpu:resize(3, batchSize, opt.nChannels, 64, 64)
      X_gpu:resize(3, batchSize, opt.nChannels, 64, 64)
      Sid_cpu:resize(3, batchSize):random(1, opt.nClasses)
      Sid_cpu[2]:copy(Sid_cpu[1])
      Sid_gpu:resize(3, batchSize)
      for i = 1, 3 do
	 for j = 1, batchSize do
	    local data_class = data[set][Sid_cpu[i][j] ]
	    if opt.dataset == 'yaleB' then
	       X_cpu[i][j]:copy(data_class[torch.random(data_class:size(1))])
	    elseif opt.dataset == 'yaleBExtended' then
	       local startidx = torch.random(17)
	       X_cpu[i][j]:copy(data_class[torch.random(data_class:size(1))]:narrow(3, startidx, 64))
	    end
	 end
      end	 
      X_gpu:copy(X_cpu)
      Sid_gpu:copy(Sid_cpu)
      local X_gpu_ds = downscaler:forward(X_gpu:view(3*batchSize, opt.nChannels, 64, 64))
      return X_gpu_ds:view(3, batchSize, opt.nChannels, opt.h, opt.w), Sid_gpu
   end
end

function getPubfig_getBatch(opt)
   local datapath = opt.datapath .. '/pubfig/'
   opt.nChannels = 3
   opt.h = opt.h or 64
   opt.w = opt.w or 64
   local sets = {}
   local index = torch.load(datapath .. 'cropped_64x64.t7')
   local data = {train = index.files.eval, test = index.files.dev}
   local labels = {train = index.people.eval, test = index.people.dev}
   opt.nClasses = #labels.train
   for k, v in pairs(data) do
      for k2,v2 in pairs(v) do
	 v[k2] = v2:type(torch.getdefaulttensortype()):mul(2):add(-1):clamp(-1,1)
	 --print(v[k2]:min())
      end
   end

   local downscaler = nn.Identity():cuda()
   if opt.h ~= 64 then
      assert((opt.h == opt.w) and (64 % opt.h == 0), 'not implemented')
      require 'cunn'
      local p = 64/opt.h
      downscaler = nn.SpatialAveragePooling(p, p, p, p):cuda()
   end
   
   return function(batchSize, set)
      batchSize = batchSize or opt.batchSize
      assert(data[set] ~= nil, 'Unknown set ' .. set)
      X_cpu:resize(3, batchSize, opt.nChannels, 64, 64)
      X_gpu:resize(3, batchSize, opt.nChannels, 64, 64)
      Sid_cpu:resize(3, batchSize):random(1, opt.nClasses)
      Sid_cpu[2]:copy(Sid_cpu[1])
      Sid_gpu:resize(3, batchSize)
      for i = 1, 3 do
	 for j = 1, batchSize do
	    local data_class = data[set][labels[set][Sid_cpu[i][j]] ]
	    X_cpu[i][j]:copy(data_class[torch.random(data_class:size(1))])
	 end
      end	 
      X_gpu:copy(X_cpu)
      Sid_gpu:copy(Sid_cpu)
      local X_gpu_ds = downscaler:forward(X_gpu:view(3*batchSize, opt.nChannels, 64, 64))
      return X_gpu_ds:view(3, batchSize, opt.nChannels, opt.h, opt.w), Sid_gpu
   end
end

function get_getBatch(opt)
   if opt.dataset == 'mnist' then
      return getMnist_getBatch(opt)
   elseif opt.dataset == 'sprites' then
      return getSprites_getBatch(opt)
   elseif (opt.dataset == 'yaleB') or (opt.dataset == 'yaleBExtended') then
      return getYaleB_getBatch(opt)
   elseif opt.dataset == 'pubfig' then
      return getPubfig_getBatch(opt)
   elseif opt.dataset == 'norb' then
      error("dataset not implemented yet")
   elseif opt.dataset == 'celebA' then
      error("dataset not implemented yet")
   else
      error('Unknown dataset ' .. opt.dataset)
   end
end