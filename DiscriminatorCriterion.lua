
local DiscriminatorCriterion, parent = torch.class('DiscriminatorCriterion', 'nn.Criterion') -- heritage en torch

function DiscriminatorCriterion:__init() -- constructeur
   -- equivalent a parent.__init(self)
   self.gradInputReal = torch.Tensor()
   self.gradInputFake = torch.Tensor()
   self.output = 0
end

function DiscriminatorCriterion:forward(input)
   return self:updateOutput(input)
end


function DiscriminatorCriterion:updateOutput(input)
   assert(input:nDimension() == 2, 'Datas should be of dimension 2')

   local batch_size = input:size()[1]/2
   self.output = 0
   self.output = torch.sum(torch.log(input:sub(1, batch_size)) + torch.log(1 - input:sub(batch_size+1, 2*batch_size)))

   return self.output/batch_size
end

function DiscriminatorCriterion:backward(input)
   return self:updateGradInput(input)
end

function DiscriminatorCriterion:updateGradInput(input)
   assert(input:nDimension() == 2, 'Datas should be of dimension 2')

   local batch_size = input:size()[1]/2

   self.gradInputReal = torch.Tensor(batch_size,1)
   self.gradInputFake = torch.Tensor(batch_size,1)

   self.gradInputReal = torch.cdiv(torch.ones(batch_size,1), input:sub(1,batch_size))
   self.gradInputFake = -torch.cdiv(torch.ones(batch_size,1), 1 - input:sub(batch_size+1,2*batch_size))

   return torch.cat(self.gradInputReal, self.gradInputFake, 1)
end
