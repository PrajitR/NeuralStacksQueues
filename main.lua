local Controller = require 'controller'
local Utils = require 'utils'
local Task = require 'task'

local opt = { 
    dropout = 0.0,
    num_lstm_layers = 2,
    num_memory_modules = 3,
    vocab_size = 20,
    rnn_size = 3,
    memory_size = 2,
    batch_size = 2,
    memory_types = "sqs",
    use_deque = false,
    min_train_seq_len = 1,
    max_train_seq_len = 2,
    min_test_seq_len = 3,
    max_test_seq_len = 4,
    max_grad_norm = 1
}

local protos = {}
if opt.use_deque then
    protos.rnn = Controller.DeQue(opt)
else
    protos.rnn = Controller.oneSidedMemory(opt)
end
protos.criterion = nn.ClassNLLCriterion()
Utils.gpu(protos.rnn, opt)
Utils.gpu(protos.criterion, opt)

local params, grad_params = Utils.combineAllParameters(protos.rnn)

local clones = {}
for name, proto in pairs(protos) do
    clones[name] = Utils.cloneManyTimes(proto, 2 * opt.max_test_seq_len + 2) 
end

local init_state = {}
for i = 1, opt.num_lstm_layers do
    table.insert(init_state, Utils.gpu(torch.zeros(opt.batch_size, opt.rnn_size), opt))
    table.insert(init_state, Utils.gpu(torch.zeros(opt.batch_size, opt.rnn_size), opt))
end
for i = 1, opt.num_memory_modules do
    table.insert(init_state, Utils.gpu(torch.zeros(opt.batch_size, 1, opt.memory_size), opt))
    table.insert(init_state, Utils.gpu(torch.zeros(opt.batch_size, 1), opt))
    table.insert(init_state, Utils.gpu(torch.zeros(opt.batch_size, opt.memory_size), opt))
    if opt.use_deque then
        table.insert(init_state, Utils.gpu(torch.zeros(opt.batch_size, opt.memory_size), opt))
    end
end

local function forward(x, seq_len)
    local rnn_states = {[0] = init_state} -- The rnn state going into timestep t.
    local predictions = {}

    -- Feed in input sequence, including start token.
    for t = 0, seq_len do
        local inp = t == 0 and Task.startToken(opt) or x[{{}, t}]
        output = clones.rnn[t]:forward{inp, unpack(rnn_states[t])}
        rnn_states[t + 1] = Utils.spliceList(output, 1, #init_state)
    end

    -- Feed in output sequence, including separation and end token.
    local sep_index = seq_len + 1
    local loss = 0
    for t = 0, seq_len do
        local inp = t == 0 and Task.sepToken(opt) or Task.copy(x, t)
        output = clones.rnn[t + sep_index]:forward{inp, unpack(rnn_states[t + sep_index])}
        rnn_states[t + sep_index + 1] = Utils.spliceList(output, 1, #init_state)
        predictions[t] = output[#output]
        local y = t == seq_len and Task.endToken(opt) or Task.copy(x, t + 1)
        loss = loss + clones.criterion[t + sep_index]:forward(predictions[t], y)
    end

    return rnn_states, predictions, loss
end

local function backward(x, seq_len, rnn_states, predictions)
    local d_state = Utils.spliceList(rnn_states[#rnn_states], 1, #init_state, true)   

    -- Backward pass over output sequence.
    local sep_index = seq_len + 1
    for t = seq_len, 0, -1 do
        local y = t == seq_len and Task.endToken(opt) or Task.copy(x, t + 1)
        local d_loss = clones.criterion[t + sep_index]:backward(predictions[t], y)
        table.insert(d_state, d_loss)
        local inp = t == 0 and Task.sepToken(opt) or Task.copy(x, t)
        local d_timestep = clones.rnn[t + sep_index]:backward(
            {inp, unpack(rnn_states[t + sep_index])}, d_state) 
        d_state = Utils.spliceList(d_timestep, 2, #d_timestep)
    end

    -- Backward pass over input sequence.
    local zero_crit = Utils.gpu(torch.zeros(predictions[1]:size()))
    for t = seq_len, 0, -1 do
        table.insert(d_state, zero_crit)
        local inp = t == 0 and Task.startToken(opt) or x[{{}, t}]
        local d_timestep = clones.rnn[t]:backward(
            {inp, unpack(rnn_states[t])}, d_state) 
        d_state = Utils.spliceList(d_timestep, 2, #d_timestep)
    end 
end

local function feval(paramsx)
    if params ~= paramsx then
        params:copy(paramsx)
    end
    grad_params:zero()

    local x, seq_len = Task.generateSequence(false, false, opt)
    x = Utils.gpu(x, opt)
    local rnn_states, predictions, loss = forward(x, seq_len) 
    backward(x, seq_len, rnn_states, predictions)

    local grad_norm = grad_params:norm()
    if grad_norm > opt.max_grad_norm then
        local scaling_factor = opt.max_grad_norm / grad_norm
        grad_params:mul(scaling_factor)
    end 

    return loss, grad_params
end

feval(params)
