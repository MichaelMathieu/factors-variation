-- Based on JoinTable module

require 'nn'

local Reparametrize, parent = torch.class('nn.Reparametrize', 'nn.Module')

function Reparametrize:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.output = torch.CudaTensor()
    self.gradInput = {torch.CudaTensor(), torch.CudaTensor()}
    self.eps = torch.CudaTensor()
end 

function Reparametrize:clearState()
   parent:clearState()
   self.gradInput = {torch.CudaTensor(), torch.CudaTensor()}
end

function Reparametrize:updateOutput(input)
    --Different eps for whole batch, or one and broadcast?

   self.eps:resizeAs(input[2]):normal()
   self.output:resizeAs(input[2]):mul(input[2], 0.5):exp():cmul(self.eps):add(input[1])
   --self.eps = input[2]:clone():normal()
   --self.eps = torch.randn(input[2]:size(1),self.dimension)
   --self.output = torch.mul(input[2],0.5):exp():cmul(self.eps)
   --self.output:add(input[1])

   return self.output
end

function Reparametrize:updateGradInput(input, gradOutput)
    -- Derivative with respect to mean is 1
   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)
   --self.gradInput[1] = gradOutput:clone()
    
   --Not sure if this gradient is right -- Michael: I think it is
   self.gradInput[2]:resizeAs(input[2]):mul(input[2], 0.5):exp():cmul(self.eps):mul(0.5):cmul(gradOutput)
   --self.gradInput[2] = torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps)
   --self.gradInput[2]:cmul(gradOutput)

   return self.gradInput
end
