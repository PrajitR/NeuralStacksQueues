local Controller = require 'controller'
local Utils = require 'utils'
local Task = require 'task'

local opt = { 
    dropout = 0.0,
    num_lstm_layers = 2,
    num_memory_modules = 3,
    vocab_size = 20,
    rnn_size = 3,
    embedding_size = 3,
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
protos.enc_dict = nn.LookupTable(opt.vocab_size, opt.embedding_size)
protos.dec_dict = nn.LookupTable(opt.vocab_size, opt.embedding_size)
for _, proto in pairs(protos) do
    Utils.gpu(proto, opt)
end

local params, grad_params = Utils.combineAllParameters(
    protos.rnn, protos.enc_dict, protos.dec_dict)

local clones = {}
clones.rnn = Utils.cloneManyTimes(protos.rnn, 2 * opt.max_test_seq_len + 2)
clones.criterion = Utils.cloneManyTimes(protos.criterion, opt.max_test_seq_len + 1)
clones.enc_dict = Utils.cloneManyTimes(protos.enc_dict, opt.max_test_seq_len + 1)
clones.dec_dict = Utils.cloneManyTimes(protos.dec_dict, opt.max_test_seq_len + 1)

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

local function determineInput(inp, prediction, get_argmax)
    if prediction == nil or not get_argmax then
        return inp
    else 
        local _, argmax = prediction:max(2)
        return argmax:double():select(2, 1)
    end
end

local function correctMetric(no_wrongs, num_correct, y, prediction)
    local _, argmax = prediction:max(2)
    no_wrongs:cmul(y:eq(argmax:double()))
    num_correct:add(no_wrongs:double())
end

local function forward(x, seq_len, get_argmax)
    local rnn_states = {[0] = init_state} -- The rnn state going into timestep t.
    local embeddings = {}

    -- Feed in input sequence, including start token.
    for t = 0, seq_len do
        local inp = t == 0 and Task.startToken(opt) or x:select(2, t)
        embeddings[t] = clones.enc_dict[t]:forward(inp)
        output = clones.rnn[t]:forward{embeddings[t], unpack(rnn_states[t])}
        rnn_states[t + 1] = Utils.spliceList(output, 1, #init_state)
    end

    -- Feed in output sequence, including separation and end token.
    local dec_inputs = {}
    local predictions = {}
    local sep_index = seq_len + 1
    local loss = 0
    -- Additional metrics for coarse and fine scores if testing.
    local no_wrongs, num_correct
    if get_argmax then
        no_wrongs = torch.ByteTensor(opt.batch_size):fill(1)
        num_correct = torch.zeros(opt.batch_size)
    end
 
    for t = 0, seq_len do
        local inp = t == 0 and Task.sepToken(opt) or Task.copy(x, t)
        dec_inputs[t] = determineInput(inp, predictions[t - 1], get_argmax)
        embeddings[t + seq_len] = clones.dec_dict[t]:forward(dec_inputs[t])
        output = clones.rnn[t + sep_index]:forward{
                embeddings[t + seq_len], unpack(rnn_states[t + sep_index]) }

        rnn_states[t + sep_index + 1] = Utils.spliceList(output, 1, #init_state)
        predictions[t] = output[#output]
        local y = t == seq_len and Task.endToken(opt) or Task.copy(x, t + 1)
        loss = loss + clones.criterion[t]:forward(predictions[t], y)
        if get_argmax then
            correctMetric(no_wrongs, num_correct, y, predictions[t])
        end
    end

    return rnn_states, predictions, dec_inputs, embeddings, loss, num_correct
end

local function backward(x, seq_len, rnn_states, dec_inputs, embeddings, predictions)
    local d_state = Utils.spliceList(rnn_states[#rnn_states], 1, #init_state, true)   

    -- Backward pass over output sequence.
    local sep_index = seq_len + 1
    for t = seq_len, 0, -1 do
        local y = t == seq_len and Task.endToken(opt) or Task.copy(x, t + 1)
        local d_loss = clones.criterion[t]:backward(predictions[t], y)
        table.insert(d_state, d_loss)
        local d_timestep = clones.rnn[t + sep_index]:backward(
            { embeddings[t + sep_index], unpack(rnn_states[t + sep_index]) }, d_state) 
        clones.dec_dict[t]:backward(dec_inputs[t], d_timestep[1])
        d_state = Utils.spliceList(d_timestep, 2, #d_timestep)
    end

    -- Backward pass over input sequence.
    local zero_crit = Utils.gpu(torch.zeros(predictions[1]:size()))
    for t = seq_len, 0, -1 do
        table.insert(d_state, zero_crit)
        local d_timestep = clones.rnn[t]:backward(
            { embeddings[t], unpack(rnn_states[t]) }, d_state) 
        local inp = t == 0 and Task.startToken(opt) or x:select(2, t)
        clones.enc_dict[t]:backward(inp, d_timestep[1])
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
    local rnn_states, predictions, dec_inputs, embeddings, loss = forward(x, seq_len, true) 
    backward(x, seq_len, rnn_states, dec_inputs, embeddings, predictions)

    local grad_norm = grad_params:norm()
    if grad_norm > opt.max_grad_norm then
        local scaling_factor = opt.max_grad_norm / grad_norm
        grad_params:mul(scaling_factor)
    end 

    return loss, grad_params
end

feval(params)
