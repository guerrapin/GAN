
local GeneratorCriterion, parent = torch.class('GeneratorCriterion', 'nn.Criterion') -- heritage en torch

function GeneratorCriterion:__init() -- constructeur
   -- equivalent a parent.__init(self)
   self.gradInput = torch.Tensor()
   self.output = 0
end

function GeneratorCriterion:forward(input) -- appel generique pour calculer le cout
   return self:updateOutput(input)
end


function GeneratorCriterion:updateOutput(input)
   assert(input:nDimension() == 2, 'Datas should be of dimension 2')

   local n_batch = input:size()[1]
   self.output = 0

   self.output = torch.sum(torch.log(input))

   return self.output/n_batch
end

function GeneratorCriterion:backward(input) -- appel generique pour calculer le gradient du cout
   return self:updateGradInput(input)
end

function GeneratorCriterion:updateGradInput(input)

   n_batch = input:size()[1]

   self.gradInput = torch.Tensor(n_batch,1)

   self.gradInput = torch.cdiv(torch.ones(n_batch,1), input)

   return self.gradInput
end
