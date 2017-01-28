require 'gnuplot'
require 'nn'
require 'optim'

require 'DiscriminatorCriterion'
require 'GeneratorCriterion'

require 'utils'
require 'csvigo'

---------------------------------------------------------
------------- COMMAND OPTIONS ---------------------------
---------------------------------------------------------

cmd = torch.CmdLine()
------------- Algorithm ------------
cmd:option('-batch_size',40,"mini-batch size")
cmd:option('-maxEpoch',5000,"number of epochs")
cmd:option('-learning_rate',1e-4,"learning rate")
cmd:option('-k', 1, "number of discriminator training iteration for one generative training iteration")
cmd:option('-seed_value', 1010, "seed value for random generated data")
cmd:option('-lr_decay_start', 500, "iteration when to start decaying the learning rate (0 = No decay)")
cmd:option('-lr_decay_every', 100,"every how many iteration thereafter to drop LR by half? ")
cmd:option('-plot', true, "plot or not the data point while training")
cmd:option('-ratio', 0.8, "train_data/all_data ratio")

------------ Model -----------------
cmd:option('-noise_size', 100, "dimension of the noise vector")
cmd:option('-noise_type', "Gaussian", "either Gaussian or Uniform")
cmd:option('-noise_mean',4 , "mean value for the noise distribution")
cmd:option('-noise_var', 0.5, "variance for the noise distribution")
cmd:option('-generative_size', 500, "dimension of the hidden layers of the generative model")
cmd:option('-discrim_size', 200, "dimension of the hidden layers of the discriminant model")

------------ Data ------------------
cmd:option('-mnist_size',"big","either big (28x28) or small (8x8)")
--cmd:option('-mnist_label',8,"label of the images to consider, 10 if all labels")

local opt = cmd:parse(arg)
print("GAN Implementation with Gaussian distributed data")
print("Parameters of this experiment :")
print(opt)


-- Loggers
dloss_logger = optim.Logger('dloss.log') -- loss with the DiscriminatorCriterion
dlogger_fake = optim.Logger('dfake.log') --  output prediction on fake data
dlogger_real = optim.Logger('dreal.log') -- output prediction on real data
gloss_logger = optim.Logger('gloss.log') -- loss with the GeneratorCriterion

dlogger_fake:setNames{'Discriminator output on fake data'}
dlogger_real:setNames{'Discriminator output on real data'}
gloss_logger:setNames{'Generator Loss'}; dloss_logger:setNames{'Discriminator loss'}
gloss_logger:style{'+-'}; dloss_logger:style{'+-'}; dlogger_fake:style{'+-'}; dlogger_real:style{'+-'}

torch.manualSeed(opt.seed_value)

---------------------------------------------------------
-------------- DATA GENERATION --------------------------
---------------------------------------------------------


if opt.mnist_size == "big" then
   local mnist = require 'mnist'
   local train = mnist.traindataset()
   local test = mnist.testdataset()

   xs_train = torch.Tensor(5851,28*28)
   xs_test = torch.Tensor(974,28*28)

   iterator = 1
   for i = 1, train.data:size(1) do
      if train.label[i] == 8 then
         xs_train[iterator] = torch.reshape(train.data[i],28*28):double()
         iterator = iterator +1
      end
   end
   iterator = 1
   for i = 1, test.data:size(1) do
      if test.label[i] == 8 then
         xs_test[iterator] = torch.reshape(test.data[i],28*28):double()
         iterator = iterator +1
      end
   end

   --local xs_train = torch.div(torch.reshape(train.data,6000,28*28):double(),torch.max(train.data))
   --local xs_test = torch.div(torch.reshape(test.data,1000,28*28):double(),torch.max(train.data))

   dimension = xs_train:size(2)
   train_size = xs_train:size(1)
   test_size = xs_test:size(1)

else
   data_table = csvigo.load({path = "mnist_8x8_8.csv", mode = "large"})

   dimension = #data_table[1]
   train_size = torch.floor(#data_table*opt.ratio)
   test_size = #data_table - train_size

   xs_train = torch.Tensor(train_size, dimension)
   xs_test = torch.Tensor(test_size, dimension)

   for i =1,#data_table do
      for j = 1, dimension do
         if i<=train_size then
            xs_train[i][j] = data_table[i][j]
         else
            xs_test[i-train_size][j] = data_table[i][j]
         end
      end
   end
end

-- Normalisation of the pixel points
-- (données centrées réduites)

data_var = torch.var(xs_train,1)
data_mean = torch.mean(xs_train,1)

for i = 1,train_size do
   for j = 1, dimension do
      xs_train[i][j] = xs_train[i][j] - data_mean[1][j]
      --if data_var[1][j] ~= 0 then
      --   xs_train[i][j] = xs_train[i][j]/data_var[1][j]
      --end
   end
end

for i = 1,test_size do
   for j = 1, dimension do
      xs_test[i][j] = xs_test[i][j] - data_mean[1][j]
      --if data_var[1][j] ~= 0 then
      --   xs_test[i][j] = xs_test[i][j]/data_var[1][j]
      --end
   end
end

----------------------------------------------------
-------------- MODEL CONSTRUCTION ------------------
----------------------------------------------------

-- Create the neural networks

local Generator = nn.Sequential()
Generator:add(nn.Linear(opt.noise_size, opt.generative_size)):add(nn.ReLU())
Generator:add(nn.Linear(opt.generative_size, opt.generative_size)):add(nn.ReLU())
Generator:add(nn.Linear(opt.generative_size, dimension))

local Discriminator = nn.Sequential()
Discriminator:add(nn.Linear(dimension, opt.discrim_size)):add(nn.ReLU())
Discriminator:add(nn.Linear(opt.discrim_size, opt.discrim_size)):add(nn.ReLU())
Discriminator:add(nn.Linear(opt.discrim_size, 1)):add(nn.Sigmoid())

local Discrim_criterion = DiscriminatorCriterion()
local Gen_criterion = GeneratorCriterion()

---------------------------------------------------------
-------------- LEARNING AND EVALUATION ------------------
---------------------------------------------------------

function Eval(iteration)
   -- to log losses and decisions and display distributions

   -- test noise creation
   local test_noise_z
   if opt.noise_type == "Gaussian" then
      test_noise_z = torch.randn(test_size, opt.noise_size)*opt.noise_var + opt.noise_mean
   else
      test_noise_z = torch.rand(test_size, opt.noise_size)*opt.noise_var + opt.noise_mean
   end

   -- generate data according to Pg
   local fake_data = Generator:forward(test_noise_z)

   -- Concatenate real and fake data
   local all_data = torch.cat(xs_test, fake_data, 1)

   -- Compute Discriminator decision
   local decision = Discriminator:forward(all_data)

   -- to log decision on real and fake data
   dlogger_fake:add{torch.mean(decision:sub(test_size+1,2*test_size))}
   dlogger_real:add{torch.mean(decision:sub(1,test_size))}

   -- compute Loss of Discriminator
   local discrim_loss = Discrim_criterion:forward(decision)
   dloss_logger:add{discrim_loss}

   local gen_loss = Gen_criterion:forward(decision:sub(test_size+1, 2*test_size))
   gloss_logger:add{gen_loss}

   -- displaying stuff
   if iteration%100 == 0 then
      print("Achievement : " .. iteration/opt.maxEpoch*100 .. "%")
   end

   if iteration%1 == 0 then
      data_display = fake_data[1]
      for m = 1, dimension do
         --if data_var[1][m] ~=0 then
         --   data_display[m] = data_display[m]*data_var[1][m]
         --end
         data_display[m] = data_display[m] + data_mean[1][m]
      end
      if opt.mnist_size == "big" then
         gnuplot.imagesc(torch.reshape(data_display, 28,28))
      else
         gnuplot.imagesc(torch.reshape(data_display, 8,8))
      end
   end
end


local iterator = 1

-- loop over the epochs
for iteration=1,opt.maxEpoch do

   -- displaying stuff
   Eval(iteration)

   -- learning rate decay stuff
   if iteration > opt.lr_decay_start and opt.lr_decay_start >= 0 then
      if iteration % opt.lr_decay_every == 0 then
         opt.learning_rate = opt.learning_rate * 0.5
      end
   end

   local next_epoch = false
   local shuffle = torch.randperm(train_size)

   while next_epoch == false do

      -- Discriminator optimisation

      for step = 1, opt.k do

         -- sample minibatch of noise samples from noise prior (Gaussian noise)
         local noise_z
         if opt.noise_type == "Gaussian" then
            noise_z = torch.randn(opt.batch_size, opt.noise_size)*opt.noise_var + opt.noise_mean
         else
            noise_z = torch.rand(opt.batch_size, opt.noise_size)*opt.noise_var + opt.noise_mean
         end

         -- sample minibatch of examples from data distribution
         local real_data = torch.Tensor(opt.batch_size, dimension)
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

         -- Generate data from Generator
         local fake_data = Generator:forward(noise_z)

         -- Concatenate real and fake data
         local all_data = torch.cat(real_data, fake_data, 1)

         -- Compute Discriminator decision
         local decision = Discriminator:forward(all_data)

         -- to log decision on real and fake data
         -- dlogger_fake:add{torch.mean(decision:sub(opt.batch_size+1,2*opt.batch_size))}
         -- dlogger_real:add{torch.mean(decision:sub(1,opt.batch_size))}

         -- compute Loss of Discriminator
         -- local discrim_loss = Discrim_criterion:forward(decision)
         -- dloss_logger:add{discrim_loss}

         -- backward deltas on decision then backward on the discriminator
         local decision_delta = Discrim_criterion:backward(decision)
         Discriminator:backward(all_data, decision_delta)

         -- update discriminator
         Discriminator:updateParameters(- opt.learning_rate) -- minus because it's a maximisation problem

      end

      -- Generator Optimisation

      -- sample minibatch of noise samples from noise prior (Gaussian noise)
      local noise_z
      if opt.noise_type == "Gaussian" then
         noise_z = torch.randn(opt.batch_size, opt.noise_size)*opt.noise_var + opt.noise_mean
      else
         noise_z = torch.rand(opt.batch_size, opt.noise_size)*opt.noise_var + opt.noise_mean
      end

      -- to be sure that accumulated gradients are deleted
      Generator:zeroGradParameters()
      Discriminator:zeroGradParameters()

      -- forward generator and discriminator
      local fake_data = Generator:forward(noise_z)
      local decision_fake = Discriminator:forward(fake_data)

      -- local gen_loss = Gen_criterion:forward(decision_fake)
      -- gloss_logger:add{gen_loss}

      -- compute signals of generator ouput
      local decision_fake_delta = Gen_criterion:backward(decision_fake)
      local fake_data_delta = Discriminator:backward(fake_data, decision_fake_delta)

      -- compute gradients of generator parameters
      Generator:backward(noise_z, fake_data_delta)

      Discriminator:zeroGradParameters() -- because we don't want those fake data to be used to update Discriminator parameters

      -- update generator
      Generator:updateParameters(- opt.learning_rate) -- minus because it's a maximisation problem
   end

end

gloss_logger:plot()
dloss_logger:plot()
dlogger_fake:plot()
dlogger_real:plot()
