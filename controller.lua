require 'nn'
require 'nngraph'
local Memory = require 'memory'

local Controller = {}

local function LSTM(isz, rsz, x, prev_h, prev_c)
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(isz, 4 * rsz)(x)
    local h2h = nn.Linear(rsz, 4 * rsz)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rsz)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    return next_h, next_c
end

function Controller.oneSidedMemory(opt)
  -- convenience variables
  local dropout = opt.dropout or 0 
  local n = opt.num_lstm_layers
  local m = opt.num_memory_modules
  local msz = opt.memory_size
  local rsz = opt.rnn_size
  local vsz = opt.vocab_size

  -- there will be 1 + 2 * n + 3 * m inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  for s = 1, m do
    table.insert(inputs, nn.Identity()()) -- prev_memory_vectors[s]
    table.insert(inputs, nn.Identity()()) -- prev_strength[s]
    table.insert(inputs, nn.Identity()()) -- prev_read[s]
  end

  local x, input_size_L
  local outputs = {}

  for L = 1, n do
    -- cell and hidden state from previous timesteps
    local prev_h = inputs[L * 2 + 1]
    local prev_c = inputs[L * 2]

    -- the input to this layer
    if L == 1 then 
      local embedding = nn.LookupTable(vsz, rsz)(inputs[1])
      local controller_inputs = { embedding } 
      -- collect all reads from previous step
      for s = 1, m do
        table.insert(controller_inputs, inputs[2 * n + 1 + s * 3])
      end
      x = nn.JoinTable(2)(controller_inputs)
      input_size_L = rsz + m * msz
    else 
      x = outputs[(L - 1) * 2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rsz
    end

    local next_h, next_c = LSTM(input_size_L, rsz, x, prev_h, prev_c)
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  -- batch compute all controller outputs for speed
  local controller_outputs = nn.Linear(rsz, 
         vsz + m * (msz + 2))(top_h)
  local tanh_part = nn.Tanh()(
        nn.Narrow(2, 1, m * msz + vsz)(controller_outputs))
  local sigmoid_part = nn.Sigmoid()(
        nn.Narrow(2, vsz + m * msz + 1, m * 2)(controller_outputs))

  -- interact with memory
  for s = 1, m do
    local MemoryType = opt.memory_types:sub(s, s) == "q" and Memory.Queue or Memory.Stack
    local mem_index = 1 + 2 * n + (s - 1) * 3 -- get index of prev memory_vectors and strength
    local prev_memory_vectors = inputs[mem_index + 1]
    local prev_strength = inputs[mem_index + 2]
    local new_memory = nn.Narrow(2, (s - 1) * msz + 1, msz)(tanh_part)
    local pop = nn.Narrow(2, (s - 1) * 2 + 1, 1)(sigmoid_part)
    local push = nn.Narrow(2, (s - 1) * 2 + 2, 1)(sigmoid_part)

    local mem_outputs = MemoryType(prev_memory_vectors, prev_strength, 
                                   new_memory, pop, push, opt)
    for i = 1, #mem_outputs do
        table.insert(outputs, mem_outputs[i])
    end
  end

  local pred = nn.Narrow(2, m * msz + 1, vsz)(tanh_part)
  table.insert(outputs, pred)

  return nn.gModule(inputs, outputs)
end

function Controller.DeQue(opt)
  -- convenience variables
  local dropout = opt.dropout or 0 
  local n = opt.num_lstm_layers
  local m = opt.num_memory_modules
  local msz = opt.memory_size
  local rsz = opt.rnn_size
  local vsz = opt.vocab_size

  -- there will be 1 + 2 * n + 3 * m inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  for s = 1, m do
    table.insert(inputs, nn.Identity()()) -- prev_memory_vectors[s]
    table.insert(inputs, nn.Identity()()) -- prev_strength[s]
    table.insert(inputs, nn.Identity()()) -- prev_read_top[s]
    table.insert(inputs, nn.Identity()()) -- prev_read_bot[s]
  end

  local x, input_size_L
  local outputs = {}

  for L = 1, n do
    -- cell and hidden state from previous timesteps
    local prev_h = inputs[L * 2 + 1]
    local prev_c = inputs[L * 2]

    -- the input to this layer
    if L == 1 then 
      local embedding = nn.LookupTable(vsz, rsz)(inputs[1])
      local controller_inputs = { embedding } 
      -- collect all reads from previous step
      for s = 1, m do
        table.insert(controller_inputs, inputs[2 * n + 1 + s * 4 - 1]) -- read_top
        table.insert(controller_inputs, inputs[2 * n + 1 + s * 4]) -- read_bot
      end
      x = nn.JoinTable(2)(controller_inputs)
      input_size_L = rsz + m * msz * 2
    else 
      x = outputs[(L - 1) * 2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rsz
    end

    local next_h, next_c = LSTM(input_size_L, rsz, x, prev_h, prev_c)
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  -- batch compute all controller outputs for speed
  local controller_outputs = nn.Linear(rsz, 
         vsz + m * (msz + 2) * 2)(top_h)
  local tanh_part = nn.Tanh()(
        nn.Narrow(2, 1, m * msz * 2 + vsz)(controller_outputs))
  local sigmoid_part = nn.Sigmoid()(
        nn.Narrow(2, vsz + m * msz * 2 + 1, m * 4)(controller_outputs))

  -- interact with memory
  for s = 1, m do
    local mem_index = 1 + 2 * n + (s - 1) * 4 -- get index of prev memory_vectors and strength
    local prev_memory_vectors = inputs[mem_index + 1]
    local prev_strength = inputs[mem_index + 2]
    local memory_top = nn.Narrow(2, (s - 1) * msz * 2 + 1, msz)(tanh_part)
    local memory_bot = nn.Narrow(2, (s - 1) * msz * 2 + msz + 1, msz)(tanh_part)
    local pop_top = nn.Narrow(2, (s - 1) * 4 + 1, 1)(sigmoid_part)
    local pop_bot = nn.Narrow(2, (s - 1) * 4 + 2, 1)(sigmoid_part)
    local push_top = nn.Narrow(2, (s - 1) * 4 + 3, 1)(sigmoid_part)
    local push_bot = nn.Narrow(2, (s - 1) * 4 + 4, 1)(sigmoid_part)

    local mem_outputs = Memory.DeQue(prev_memory_vectors, prev_strength, 
      memory_top, memory_bot, pop_top, pop_bot, push_top, push_bot, opt)
    for i = 1, #mem_outputs do
        table.insert(outputs, mem_outputs[i])
    end
  end

  local pred = nn.Narrow(2, m * msz * 2 + 1, vsz)(tanh_part)
  table.insert(outputs, pred)

  return nn.gModule(inputs, outputs)
end

return Controller
