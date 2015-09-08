require 'optim'
require 'nn'
local Controller = require 'controller'
local Utils = require 'utils'
local Task = require 'task'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a neural transducer with memory.')
cmd:text()
cmd:text('Options:')
-- Model checkpoints.
cmd:option('-checkpoint_dir', 'checkpoints', 'directory to save checkpoints in')
cmd:option('-model_name', 'model', 'file name of model checkpoint')
cmd:option('-use_checkpoint', '', 'checkpoint name to load from')
-- Model parameters.
cmd:option('-num_lstm_layers', 1, 'number of lstm layers to use in the controller')
cmd:option('-num_memory_modules', 1, 'number of memory modules to use')
cmd:option('-rnn_size', 256, 'size of the hidden layer in the controller')
cmd:option('-memory_size', 256, 'size of individual memories')
cmd:option('-embedding_size', 64, 'size of integer encoded embeddings')
cmd:option('-vocab_size', 128, 'number of different symbols for input')
cmd:option('-use_deque', false, 'boolean indicating if a DeQue memory module is used')
cmd:option('-memory_types', 's', 'a string indicating what each memory module should be. only specify if -use_deque is not true. string should be same length as -num_memory_modules. example: "sqs" to use 2 Stacks and 1 Queue')
cmd:option('-task', 'copy', 'task to train on. "copy", "reverse", or "bigramFlip"')
-- Hyperparameters.
cmd:option('-min_train_seq_len', 8, 'minimum length of training sequence.')
cmd:option('-max_train_seq_len', 64, 'maximum length of training sequence.')
cmd:option('-min_test_seq_len', 65, 'minimum length of test sequence.')
cmd:option('-max_test_seq_len', 128, 'maximum length of test sequence.')
cmd:option('-batch_size', 10, 'size of minibatch')
cmd:option('-dropout', 0.0, 'dropout probability')
cmd:option('-learning_rate', 2e-3, 'learning rate for rmsprop')
cmd:option('-decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('-max_grad_norm', 1, 'constrain norm of gradient to be less than this')
cmd:option('-iterations_per_epoch', 100, 'number of iterations per epoch')
cmd:option('-epochs_per_validation', 10, 'number of epochs before computing accuracy')
cmd:option('-num_accuracy_batches', 100, 'number of batches for computing accuracy')
cmd:option('-seed', 1, 'random number generator seed')
-- GPU / CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
local getY = Task[opt.task]

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

local protos
if string.len(opt.use_checkpoint) > 0 then
    print('Loading checkpoint from ' .. opt.use_checkpoint)
    local checkpoint = torch.load(opt.use_checkpoint)
    protos = checkpoint.protos
    print('Overwriting model sizes to checkpoint values.')
    opt.num_lstm_layers = checkpoint.opt.num_lstm_layers
    opt.num_memory_modules = checkpoint.opt.num_memory_modules
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.memory_size = checkpoint.opt.memory_size
    opt.embedding_size = checkpoint.opt.embedding_size
    opt.batch_size = checkpoint.opt.batch_size
    opt.memory_types = checkpoint.opt.memory_types
    opt.use_deque = checkpoint.opt.use_deque
else 
    protos = {}
    if opt.use_deque then
        protos.rnn = Controller.DeQue(opt)
    else
        protos.rnn = Controller.oneSidedMemory(opt)
    end
    protos.criterion = nn.ClassNLLCriterion()
    protos.enc_dict = nn.LookupTable(opt.vocab_size, opt.embedding_size)
    protos.dec_dict = nn.LookupTable(opt.vocab_size, opt.embedding_size)
end
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
        no_wrongs = Utils.gpu(torch.ByteTensor(opt.batch_size):fill(1))
        num_correct = Utils.gpu(torch.zeros(opt.batch_size))
    end
 
    for t = 0, seq_len do
        local inp = t == 0 and Task.sepToken(opt) or getY(x, t)
        dec_inputs[t] = determineInput(inp, predictions[t - 1], get_argmax)
        embeddings[t + seq_len] = clones.dec_dict[t]:forward(dec_inputs[t])
        output = clones.rnn[t + sep_index]:forward{
                embeddings[t + seq_len], unpack(rnn_states[t + sep_index]) }

        rnn_states[t + sep_index + 1] = Utils.spliceList(output, 1, #init_state)
        predictions[t] = output[#output]
        local y = t == seq_len and Task.endToken(opt) or getY(x, t + 1)
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
        local y = t == seq_len and Task.endToken(opt) or getY(x, t + 1)
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

local function feval(params_)
    if params ~= params_ then
        params:copy(paramsx)
    end
    grad_params:zero()

    local x, seq_len = Task.generateSequence(false, false, opt)
    x = Utils.gpu(x, opt)
    local rnn_states, predictions, dec_inputs, embeddings, loss = forward(x, seq_len, true) 
    backward(x, seq_len, rnn_states, dec_inputs, embeddings, predictions)

    loss = loss / (seq_len * opt.batch_size)
    grad_params:div(seq_len * opt.batch_size)
    local grad_norm = grad_params:norm()
    if grad_norm > opt.max_grad_norm then
        local scaling_factor = opt.max_grad_norm / grad_norm
        grad_params:mul(scaling_factor)
    end 

    return loss, grad_params
end

local function accuracy(is_test)
    local coarse = 0
    local fine = 0
    for i = 1, opt.num_accuracy_batches do
        local x, seq_len = Task.generateSequence(false, is_test, opt)
        x = Utils.gpu(x)
        local _, _, _, _, _, num_correct = forward(x, seq_len, true)  
        coarse = coarse + num_correct:eq(seq_len + 1):sum()
        fine = fine + num_correct:sum() / (seq_len + 1)
    end
    local num_seqs = opt.num_accuracy_batches * opt.batch_size
    return {coarse = coarse / num_seqs, fine = fine / num_seqs}
end

local function changeRnnMode(is_training)
    for i = 0, #clones.rnn - 1 do
        if is_training then
            clones.rnn[i]:training()
        else
            clones.rnn[i]:evaluate()
        end
    end
end

local best_loss = nil
local train_losses = {}
local train_accuracy = {}
local test_accuracy = {}
local optim_state = { learningRate = opt.learning_rate, alpha = opt.decay_rate }
local converged = false
local epoch = 1

while not converged do
    local loss = 0
    changeRnnMode(true)
    local timer = torch.Timer()
    for i = 1, opt.iterations_per_epoch do
        local batch_loss = optim.rmsprop(feval, params, optim_state) 
        loss = loss + batch_loss[1]
    end
    local time = timer:time().real
    train_losses[epoch] = loss
    
    print(string.format('epoch %d, loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs', epoch, loss, grad_params:norm() / params:norm(), time))

    if epoch % opt.epochs_per_validation == 0 then
        changeRnnMode(false)
        train_accuracy[epoch] = accuracy(false) 
        test_accuracy[epoch] = accuracy(true) 
    end

    if epoch % 5 == 0 then 
        collectgarbage()
    end

    -- Save model if best loss yet.
    if best_loss == nil or loss < best_loss then
        print('Best loss yet!')
        best_loss = loss

        local savefile = string.format('%s/%s.t7', opt.checkpoint_dir, opt.model_name)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.train_accuracy = train_accuracy
        checkpoint.test_accuracy = test_accuracy
        checkpoint.epoch = epoch
        torch.save(savefile, checkpoint)
    end

    -- Convergence check
    if epoch > 5 and torch.abs(loss - train_losses[epoch - 5]) < 1e-1 then
        optim_state.learningRate = optim_state.learningRate / 2
    else if epoch > 10 and torch.abs(loss - train_losses[epoch - 10]) < 1e-3 then
        convergence = true
    end

    epoch = epoch + 1
end
