require 'nn'

local AdvCriterion, parent = torch.class('nn.AdvCriterion', 'nn.Criterion')

function AdvCriterion:__init(network, realLabel, fakeLabel)
   -- BE CAREFUL: this calls a getParameters. Meaning that
   -- every reference to the network may be invalidated.
   -- TODO allow for something else
   self.network = network:cuda()
   self.weight, self.gradWeight = self.network:getParameters()
   self.bce = nn.BCECriterion():cuda()
   self.realLabel = realLabel or 1
   self.fakeLabel = fakeLabel or 0
   self.zeros = torch.CudaTensor(1):fill(self.fakeLabel)
   self.ones = torch.CudaTensor(1):fill(self.realLabel)
   self.trainingErr, self.lastErr, self.nTotalTraining = 0, 0, 0
end

local function getBatchSize(x)
   if type(x) == 'table' then
      return getBatchSize(x[1])
   else
      return x:size(1)
   end
end

function AdvCriterion:updateOutput(input)
   local batchSize = getBatchSize(input)
   if self.ones:size(1) ~= batchSize then
      self.ones:resize(batchSize):fill(self.realLabel)
   end
   self.network:forward(input)
   self.loss = self.bce:forward(self.network.output, self.ones)
   return self.loss
end

function AdvCriterion:updateGradInput(input)
   local derr = self.bce:backward(self.network.output, self.ones)
   self.gradInput = self.network:updateGradInput(input, derr)
   return self.gradInput
end

function AdvCriterion:resetErrors()
   self.trainingErr, self.nTotalTraining = 0, 0
end

function AdvCriterion:getError()
   return self.trainingErr / self.nTotalTraining
end

function AdvCriterion:getFeval(batchNatural, batchGenerated)
   local batchSize = getBatchSize(batchNatural)
   if self.ones:size(1) ~= batchSize then
      self.ones:resize(batchSize):fill(self.realLabel)
   end
   if self.zeros:size(1) ~= batchSize then
      self.zeros:resize(batchSize):fill(self.fakeLabel)
   end
   return function(params)
      assert(params == self.weight)
      self.gradWeight:zero()
      local outputN = self.network:forward(batchNatural)
      local errN = self.bce:forward(outputN, self.ones)
      local derrN = self.bce:backward(outputN, self.ones)
      self.network:backward(batchNatural, derrN)
      local outputG = self.network:forward(batchGenerated)
      local errG = self.bce:forward(outputG, self.zeros)
      local derrG = self.bce:backward(outputG, self.zeros)
      self.network:backward(batchGenerated, derrG)
      self.lastErr = (errN + errG) / 2
      self.trainingErr = self.trainingErr + self.lastErr
      self.nTotalTraining = self.nTotalTraining + 1
      return errN + errG, self.gradWeight
   end
end