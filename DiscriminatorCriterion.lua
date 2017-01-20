
local DiscriminatorCriterion, parent = torch.class('DiscriminatorCriterion', 'nn.Criterion') -- heritage en torch

function DiscriminatorCriterion:__init() -- constructeur
   -- equivalent a parent.__init(self)
   self.gradInputReal = torch.Tensor()
   self.gradInputFake = torch.Tensor()
   self.output = 0
end

function DiscriminatorCriterion:forward(inputReal, inputFake)
   return self:updateOutput(inputReal, inputFake)
end


function DiscriminatorCriterion:updateOutput(inputReal, inputFake)
   assert(inputReal:nDimension() == inputFake:nDimension(), 'number of dimensions mismatch')
   assert(inputReal:nDimension() == 2, 'Datas should be of dimension 2')

   assert(inputReal:size()[1] == inputFake:size()[1], "mini-batch size mismatch")
   assert(inputReal:size()[2] == inputFake:size()[2], "dimension size mismatch")

   local n_batch = inputReal:size()[1]
   self.output = 0

   self.output = torch.sum(torch.log(inputReal) + torch.log(1 - inputFake))

   return self.output/n_batch
end

function DiscriminatorCriterion:backward(inputReal, inputFake)
   return self:updateGradInput(inputReal, inputFake)
end

function DiscriminatorCriterion:updateGradInput(inputReal, inputFake)
   assert(inputReal:nDimension() == inputFake:nDimension(), 'number of dimensions mismatch')

   assert(inputReal:size()[1] == inputFake:size()[1], "mini-batch size mismatch")
   assert(inputReal:size()[2] == inputFake:size()[2], "data dimension size mismatch")

   n_batch = inputReal:size()[1]

   self.gradInputReal = torch.Tensor(n_batch,1)
   self.gradInputFake = torch.Tensor(n_batch,1)

   self.gradInputReal = torch.cdiv(torch.ones(n_batch,1), inputReal)
   self.gradInputFake = torch.cdiv(torch.ones(n_batch,1), 1 - inputFake)

   return self.gradInputReal, self.gradInputFake
end
