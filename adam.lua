-- from A. Karpathy (neuraltalk2 project) : https://github.com/karpathy/neuraltalk2

function adam(x, dx, lr, beta1, beta2, epsilon, state)
  local beta1 = beta1 or 0.9
  local beta2 = beta2 or 0.999
  local epsilon = epsilon or 1e-8

  if not state.m then
    -- Initialization
    state.t = 0
    -- Exponential moving average of gradient values
    state.m = x.new(#dx):zero()
    -- Exponential moving average of squared gradient values
    state.v = x.new(#dx):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.tmp = x.new(#dx):zero()
  end

  -- Decay the first and second moment running average coefficient
  state.m:mul(beta1):add(1-beta1, dx)
  state.v:mul(beta2):addcmul(1-beta2, dx, dx)
  state.tmp:copy(state.v):sqrt():add(epsilon)

  state.t = state.t + 1
  local biasCorrection1 = 1 - beta1^state.t
  local biasCorrection2 = 1 - beta2^state.t
  local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
  
  -- perform update
  x:addcdiv(-stepSize, state.m, state.tmp)
end
