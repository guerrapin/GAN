require 'gnuplot'
require 'nn'
require 'optim'

require 'DiscriminatorCriterion'
require 'GeneratorCriterion'

local utils = require 'utils' -- few external functions

---------------------------------------------------------
------------- COMMAND OPTIONS ---------------------------
---------------------------------------------------------

cmd = torch.CmdLine()
------------- Algorithm ------------
cmd:option('-batch_size',100,"mini-batch size")
cmd:option('-maxEpoch',10,"number of epochs")
cmd:option('-learning_rate',1e-5,"learning rate")
cmd:option('-k', 1, "number of discriminator training iteration for one generative training iteration")

------------- Data -----------------
cmd:option('-dimension', 2, "dimension of the example data")
cmd:option('-n_points', 1000, "number of examples")
cmd:option('-ratio',0.8,"train/total ratio. To split the dataset in train and test sets")
cmd:option('-mean', 2, "mean of the Gaussian distribution to sample from")
cmd:option('-var', 0.4, "variance of the Gaussian distribution to sample from")

------------ Model -----------------
cmd:option('-noise_size', 2, "dimension of the noise vector")
cmd:option('-noise_type', "Gaussian", "either Gaussian or Uniform")
cmd:option('-generative_size', 120, "dimension of the hidden layers of the generative model")
cmd:option('-discrim_size', 64, "dimension of the hidden layers of the discriminant model")

local opt = cmd:parse(arg)
print("GAN Implementation with Gaussian distributed data")
print("Parameters of this experiment :")
print(opt)

-- Loggers
dloss_logger = optim.Logger('dloss.log')
gloss_logger = optim.Logger('gloss.log')
gloss_logger:setNames{'Generator Loss'}; dloss_logger:setNames{'Discriminator loss'}
gloss_logger:style{'+-'}; dloss_logger:style{'+-'}

---------------------------------------------------------
-------------- DATA GENERATION --------------------------
---------------------------------------------------------

-- Gaussian sampling

local mean = torch.Tensor(opt.dimension):fill(opt.mean)
local xs = torch.Tensor(opt.n_points, opt.dimension)

for i = 1, opt.n_points do
   xs[i]:copy(torch.randn(opt.dimension)*opt.var + mean)
end

-- Split the data in train and test sets

local train_size = torch.floor(opt.n_points*opt.ratio)

local xs_train = torch.Tensor(train_size, opt.dimension)
local xs_test = torch.Tensor(opt.n_points - train_size, opt.dimension)

for i = 1, opt.n_points do
   if i <= train_size then
      xs_train[i]:copy(xs[i])
   else
      local idx_test = i-train_size
      xs_test[idx_test]:copy(xs[i])
   end
end

----------------------------------------------------
-------------- MODEL CONSTRUCTION ------------------
----------------------------------------------------

-- Create the neural networks

local Generator = nn.Sequential()
Generator:add(nn.Linear(opt.noise_size, opt.generative_size)):add(nn.ReLU())
Generator:add(nn.Linear(opt.generative_size, opt.generative_size)):add(nn.ReLU())
Generator:add(nn.Linear(opt.generative_size, opt.dimension)):add(nn.Sigmoid())

local Discriminator = nn.Sequential()
Discriminator:add(nn.Linear(opt.dimension, opt.discrim_size)):add(nn.ReLU())
Discriminator:add(nn.Linear(opt.discrim_size, opt.discrim_size)):add(nn.ReLU())
Discriminator:add(nn.Linear(opt.discrim_size, 1)):add(nn.Sigmoid())

local Discrim_criterion = DiscriminatorCriterion()
local Gen_criterion = GeneratorCriterion()

---------------------------------------------------------
-------------- LEARNING AND EVALUATION ------------------
---------------------------------------------------------

-- gnuplot.plot({xs:split(n_points/2,1)[1],'with points ls 2'},{xs:split(n_points/2,1)[2],'with points ls 1'})


function Eval()
   local test_size = xs_test:size()[1]

   -- compute the mean square error on the test set
   local noise_z
   if opt.noise_type == "Gaussian" then
      noise_z = torch.randn(test_size, opt.noise_size)
   else
      noise_z = torch.rand(test_size, opt.noise_size)
   end

   local fake_data = Generator:forward(noise_z)
   local fake_decision = Discriminator:forward(fake_data)
   local real_decision = Discriminator:forward(xs_test)

   gnuplot.plot({xs_test,"with points ls 1"},{fake_data, "with points ls 2"})

   --local mean_square_error = torch.sum(torch.pow(fake_decision, 2) + torch.pow(1 - real_decision, 2))

   --return mean_square_error/(2*test_size)
end


local iterator = 1

-- loop over the epochs
for iteration=1,opt.maxEpoch do

   local error = Eval()
   --print(error)

   if iteration%10 == 0 then
      print("Achievement : " .. iteration/opt.maxEpoch*100 .. "%")
   end

   local next_epoch = false
   local shuffle = torch.randperm(train_size)

   while next_epoch == false do


      -- Discriminator optimisation
      for step = 1, opt.k do

         -- sample minibatch of noise samples from noise prior (Gaussian noise)
         local noise_z
         if opt.noise_type == "Gaussian" then
            noise_z = torch.randn(opt.batch_size, opt.noise_size)
         else
            noise_z = torch.rand(opt.batch_size, opt.noise_size)
         end

         -- sample minibatch of examples from data distribution
         local real_data = torch.Tensor(opt.batch_size, opt.dimension)
         for index = 1,opt.batch_size do

            real_data[index] = xs_train[shuffle[iterator]]

            -- if it is the end of train set
            if iterator + 1 > train_size then
               iterator = 1
               next_epoch = true
            else
               iterator = iterator + 1
            end
         end

         Generator:zeroGradParameters()
         Discriminator:zeroGradParameters()

         local fake_data = Generator:forward(noise_z)
         local decision_fake = Discriminator:forward(fake_data)
         local decision_real = Discriminator:forward(real_data)

         local discrim_loss = Discrim_criterion:forward(decision_real, decision_fake)
         dloss_logger:add{discrim_loss}
         local decision_real_delta, decision_fake_delta = Discrim_criterion:backward(decision_real, decision_fake)

         Discriminator:backward(fake_data, decision_fake_delta)
         Discriminator:backward(real_data, decision_real_delta)

         Discriminator:updateParameters(opt.learning_rate) -- minus because it's a maximisation problem

      end

      -- Generator Optimisation

      -- sample minibatch of noise samples from noise prior (Gaussian noise)
      local noise_z
      if opt.noise_type == "Gaussian" then
         noise_z = torch.randn(opt.batch_size, opt.noise_size)
      else
         noise_z = torch.rand(opt.batch_size, opt.noise_size)
      end

      Generator:zeroGradParameters()
      Discriminator:zeroGradParameters()

      local fake_data = Generator:forward(noise_z)
      local decision_fake = Discriminator:forward(fake_data)

      local gen_loss = Gen_criterion:forward(decision_fake)
      gloss_logger:add{gen_loss}
      local decision_fake_delta = Gen_criterion:backward(decision_fake)

      local fake_data_delta = Discriminator:backward(fake_data, decision_fake_delta)
      Generator:backward(noise_z, fake_data_delta)

      Discriminator:zeroGradParameters() -- because we don't want those fake data to be used to update Discriminator parameters

      Generator:updateParameters(opt.learning_rate) -- minus because it's a maximisation problem
   end

end

--gloss_logger:plot()
--dloss_logger:plot()
