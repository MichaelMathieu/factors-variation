local KLDivCriterion, parent = torch.class('nn.KLDivCriterion', 'nn.Criterion')

function KLDivCriterion:__init( sizeAverage )
   parent.__init(self)
   
   self.sizeAverage = sizeAverage or true
   
   self.tmp = torch.CudaTensor()
   self.gradInput = {torch.CudaTensor(), torch.CudaTensor()}
end



function KLDivCriterion:updateOutput(input, target)
   --    - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
   --  where    mu = input[1]
   --        sigma = exp(input[2])

   self.tmp:resizeAs(input[1]):copy(input[2]):mul(2):exp():add(input[1]):add(-2, input[2]):add(1)
   self.output = self.tmp:sum() * 0.5
   --local KLDelement = (input[2] + 1):add(-1,torch.pow(input[1],2)):add(-1,torch.exp(input[2]))
   --self.output = - 0.5 * torch.sum(KLDelement)

    if self.sizeAverage  then
	self.output = self.output/input[1]:size(1)
    end

    return self.output
end

function KLDivCriterion:updateGradInput(input, target)
   self.gradInput[1]:resizeAs(input[1]):copy(input[1])
   self.gradInput[2]:resizeAs(input[2]):copy(input[2]):mul(2):exp():add(-1)
   --self.gradInput[1] = input[1]
   --self.gradInput[2] = -(-torch.exp(input[2])):add(1):mul(0.5) --TODO: is that correct ????

    if self.sizeAverage then
	self.gradInput[1]:div( input[1]:size(1))
	self.gradInput[2]:div( input[1]:size(1))
    end

    return self.gradInput
end
