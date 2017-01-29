require 'gnuplot'
require 'nn'
require 'optim'

require 'DiscriminatorCriterion'
require 'GeneratorCriterion'
require 'histogram'

---------------------------------------------------------
------------- COMMAND OPTIONS ---------------------------
---------------------------------------------------------

cmd = torch.CmdLine()
------------- Algorithm ------------
cmd:option('-batch_size',40,"mini-batch size")
cmd:option('-maxEpoch',2000,"number of epochs")
cmd:option('-learning_rate',1e-4,"learning rate")
cmd:option('-k', 1, "number of discriminator training iteration for one generative training iteration")
cmd:option('-seed_value', 1010, "seed value for random generated data")
cmd:option('-lr_decay_start', 1000, "iteration when to start decaying the learning rate (0 = No decay)")
cmd:option('-lr_decay_every', 100,"every how many iteration thereafter to drop LR by half? ")
cmd:option('-plot', true, "plot or not the data point while training")

------------- Data -----------------
cmd:option('-dimension', 2, "dimension of the example data")
cmd:option('-n_points', 3000, "number of examples")
cmd:option('-ratio',0.7,"train/total ratio. To split the dataset in train and test sets")
cmd:option('-mean', 8, "mean of the Gaussian distribution to sample from")
cmd:option('-var', 0.5, "variance of the Gaussian distribution to sample from")

------------ Model -----------------
cmd:option('-noise_size', 10, "dimension of the noise vector")
cmd:option('-noise_type', "Gaussian", "either Gaussian or Uniform")
cmd:option('-noise_mean',4 , "mean value for the noise distribution")
cmd:option('-noise_var', 0.5, "variance for the noise distribution")
cmd:option('-generative_size', 40, "dimension of the hidden layers of the generative model")
cmd:option('-discrim_size', 20, "dimension of the hidden layers of the discriminant model")

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

-- Gaussian sampling

local mean = torch.Tensor(opt.dimension):fill(opt.mean)
local xs = torch.Tensor(opt.n_points, opt.dimension)

for i = 1, opt.n_points do
   xs[i]:copy(torch.randn(opt.dimension)*opt.var + mean)
end

-- Split the data in train and test sets

local train_size = torch.floor(opt.n_points*opt.ratio)
local test_size = opt.n_points - train_size

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
--Generator:add(nn.Linear(opt.generative_size, opt.generative_size)):add(nn.ReLU())
Generator:add(nn.Linear(opt.generative_size, opt.dimension))

local Discriminator = nn.Sequential()
Discriminator:add(nn.Linear(opt.dimension, opt.discrim_size)):add(nn.ReLU())
--Discriminator:add(nn.Linear(opt.discrim_size, opt.discrim_size)):add(nn.ReLU())
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

   if iteration%10 == 0 then
      -- 2 dimensions display
      if opt.dimension == 2 and opt.plot == true then

         -- set axis
         gnuplot.axis({0,2*opt.mean,0,2*opt.mean})

         -- display distributions
         gnuplot.plot({xs_test,"with points ls 1"},{fake_data, "with points ls 2"})

      -- 1 D display
      elseif opt.dimension == 1 and opt.plot == true then

         -- set axis
         gnuplot.axis({-5,opt.mean*2,0,0.15})

         local n_pts = 200
         local display_decision=torch.Tensor(n_pts,2)
         for i=1,n_pts do
            display_decision[i][1]=0+(2*opt.mean-0)/n_pts*(i-1)
            display_decision[i][2]=Discriminator:forward(torch.Tensor(1):fill(display_decision[i][1]))/10
         end
         gnuplot.plot({histogram(xs_test,25),'-'},{histogram(fake_data,25),'-'},{display_decision,'-'})
      end
   end
end


local iterator = 1

-- loop over the epochs
for iteration=1,opt.maxEpoch do

   -- displaying stuff
   Eval(iteration)

   -- learning rate decay stuff
   if iteration >= opt.lr_decay_start and opt.lr_decay_start > 0 then
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

         -- Generate data from Generator
         local fake_data = Generator:forward(noise_z)

         -- Concatenate real and fake data
         local all_data = torch.cat(real_data, fake_data, 1)

         -- Compute Discriminator decision
         local decision = Discriminator:forward(all_data)

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
