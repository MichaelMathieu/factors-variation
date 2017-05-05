require 'cunn'
require 'cudnn'
require 'nngraph'
require 'nnx'
require 'Reparametrize'

function conv(nInput, nOutput, k, d)
   -- Returns a padded convolution with nInput/nOutput input/output planes,
   -- a k*k kernel and d*d strides. If d < 1, they are fractional
   -- strides (ie. the input is larger than the input).
   -- It is assumed than d or 1/d is integer.
   -- If the input planes are h*w pixels, the output planes are h/d by w/d.
   if (d == 1) and (k % 2 == 0) then
      print("Warning: cannot have even kernel size without stride. Increasing kernel size by 1")
      k = k + 1
   end
   if d >= 1 then
      local pad = math.ceil((k-d)/2)
      --print("conv", k,d,pad)
      return cudnn.SpatialConvolution(nInput, nOutput, k, k, d, d, pad, pad):cuda()
   else
      local pad = math.ceil((k-1/d)/2)
      local adj = 2*pad - k+1/d
      --print("fullconf",k,d,pad,adj)
      return cudnn.SpatialFullConvolution(nInput, nOutput, k, k, 1/d, 1/d,
					  pad, pad, adj, adj):cuda()
   end
end

function convblock(nInput, nOutput, k, d)
   -- Returns a block composed of
   --  a conv (see the conv function for documentation)
   --  a batch normalization
   --  a ReLU
   local block = nn.Sequential()
   block:add(conv(nInput, nOutput, k, d))
   block:add(nn.SpatialBatchNormalization(nOutput, nil, 0.1):cuda())
   block:add(cudnn.ReLU())
   return block:cuda()
end

function getMultiLookupTable(nIndex, nOutput)
   -- nIndex is a table, nOutput is a scalar
   -- expects input of size batchSize * #nIndex
   -- where input[i][j] is between 1 and nIndex[j]
   -- and outputs a tensor of size batchSize * #nIndex * nOutput
   -- where output[i][j] is lookuptable[j][input[i][j] ]
   local out = nn.Sequential()
   out:add(nn.SplitTable(2))
   local pt = nn.ParallelTable()
   out:add(pt)
   for i = 1, #nIndex do
      pt:add(nn.LookupTable(nIndex[i], nOutput):cuda())
   end
   out:add(nn.JoinTable(2))
   return out
end

function getParametersOfNetworks(...)
   local x = nn.Sequential()
   for _, y in pairs{...} do
      x:add(y)
   end
   return x:getParameters()
end

function flatBatch(x)
   local s = x:size()
   s[2] = s[1]*s[2]
   s[1] = 1
   return x:view(s):squeeze(1)
end

function unflatBatch(x, n)
   local s = x:size():totable()
   assert(s[1] % n == 0)
   table.insert(s,1,n)
   s[2] = s[2]/n
   return x:view(unpack(s))
end
