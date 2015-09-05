local Controller = require '../controller'

local tester = torch.Tester()
local tests = {}

local opt = { 
    dropout = 0.0,
    num_lstm_layers = 2,
    num_memory_modules = 3,
    vocab_size = 5,
    rnn_size = 3,
    memory_size = 2,
    batch_size = 2,
    memory_types = "sqs" }
local n_memory_vectors = 3

function tests.testOneSidedMemory()
    local rnn1 = Controller.oneSidedMemory(opt)
    local rnn2 = Controller.oneSidedMemory(opt)

    local inputs1 = {}
    local x = torch.rand(opt.batch_size):mul(opt.vocab_size):add(1):floor()
    table.insert(inputs1, x)
    for i = 1, opt.num_lstm_layers do
        table.insert(inputs1, torch.rand(opt.batch_size, opt.rnn_size))
        table.insert(inputs1, torch.rand(opt.batch_size, opt.rnn_size))
    end
    for i = 1, opt.num_memory_modules do
        table.insert(inputs1, torch.rand(opt.batch_size, n_memory_vectors, opt.memory_size))
        table.insert(inputs1, torch.rand(opt.batch_size, n_memory_vectors))
        table.insert(inputs1, torch.rand(opt.batch_size, opt.memory_size))
    end
    local output = rnn1:forward(inputs1)

    local inputs2 = {}
    table.insert(inputs2, x)
    for i = 1, #output - 1 do
        table.insert(inputs2, output[i])
    end
    rnn2:forward(inputs2)

    local grads = {}
    for i = 1, opt.num_lstm_layers do
        table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
        table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
    end
    for i = 1, opt.num_memory_modules do
        table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 2, opt.memory_size))
        table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 2))
        table.insert(grads, torch.rand(opt.batch_size, opt.memory_size))
    end
    table.insert(grads, torch.rand(opt.batch_size, opt.vocab_size))

    local gradient = rnn2:backward(inputs2, grads)
    grads = {}
    for i = 2, #gradient do 
        table.insert(grads, gradient[i])
    end
    table.insert(grads, torch.rand(opt.batch_size, opt.vocab_size))
    rnn1:backward(inputs1, grads)
end

function tests.testDeQue()
    local rnn1 = Controller.DeQue(opt)
    local rnn2 = Controller.DeQue(opt)

    local inputs1 = {}
    x = torch.rand(opt.batch_size):mul(opt.vocab_size):add(1):floor()
    table.insert(inputs1, x)
    for i = 1, opt.num_lstm_layers do
        table.insert(inputs1, torch.rand(opt.batch_size, opt.rnn_size))
        table.insert(inputs1, torch.rand(opt.batch_size, opt.rnn_size))
    end
    for i = 1, opt.num_memory_modules do
        table.insert(inputs1, torch.rand(opt.batch_size, n_memory_vectors, opt.memory_size))
        table.insert(inputs1, torch.rand(opt.batch_size, n_memory_vectors))
        table.insert(inputs1, torch.rand(opt.batch_size, opt.memory_size))
        table.insert(inputs1, torch.rand(opt.batch_size, opt.memory_size))
    end
    local output = rnn1:forward(inputs1)

    local inputs2 = {}
    table.insert(inputs2, x)
    for i = 1, #output - 1 do
        table.insert(inputs2, output[i])
    end
    rnn2:forward(inputs2)

    local grads = {}
    for i = 1, opt.num_lstm_layers do
        table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
        table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
    end
    for i = 1, opt.num_memory_modules do
        table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 4, opt.memory_size))
        table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 4))
        table.insert(grads, torch.rand(opt.batch_size, opt.memory_size))
        table.insert(grads, torch.rand(opt.batch_size, opt.memory_size))
    end
    table.insert(grads, torch.rand(opt.batch_size, opt.vocab_size))

    local gradient = rnn2:backward(inputs, grads)
    grads = {}
    for i = 2, #gradient do 
        table.insert(grads, gradient[i])
    end
    table.insert(grads, torch.rand(opt.batch_size, opt.vocab_size))
    rnn1:backward(inputs1, grads)
end

tester:add(tests)
tester:run()
