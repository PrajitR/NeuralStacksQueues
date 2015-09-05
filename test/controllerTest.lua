local Controller = require '../controller'

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

local rnn = Controller.oneSidedMemory(opt)

local inputs = {}
local x = torch.rand(opt.batch_size):mul(opt.vocab_size):add(1):floor()
table.insert(inputs, x)
for i = 1, opt.num_lstm_layers do
    table.insert(inputs, torch.rand(opt.batch_size, opt.rnn_size))
    table.insert(inputs, torch.rand(opt.batch_size, opt.rnn_size))
end
for i = 1, opt.num_memory_modules do
    table.insert(inputs, torch.rand(opt.batch_size, n_memory_vectors, opt.memory_size))
    table.insert(inputs, torch.rand(opt.batch_size, n_memory_vectors))
    table.insert(inputs, torch.rand(opt.batch_size, opt.memory_size))
end
print('\nForwards pass!')
print(unpack(rnn:forward(inputs)))

local grads = {}
for i = 1, opt.num_lstm_layers do
    table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
    table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
end
for i = 1, opt.num_memory_modules do
    table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 1, opt.memory_size))
    table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 1))
    table.insert(grads, torch.rand(opt.batch_size, opt.memory_size))
end
table.insert(grads, torch.rand(opt.batch_size, opt.vocab_size))

print('\n\nBackwards pass!')
local gradient = rnn:backward(inputs, grads)
print(unpack(gradient))
rnn:updateParameters(1)
print(#inputs)
print(#gradient)


print('\nDEQUE CONTROLLER')
rnn = Controller.DeQue(opt)

inputs = {}
x = torch.rand(opt.batch_size):mul(opt.vocab_size):add(1):floor()
table.insert(inputs, x)
for i = 1, opt.num_lstm_layers do
    table.insert(inputs, torch.rand(opt.batch_size, opt.rnn_size))
    table.insert(inputs, torch.rand(opt.batch_size, opt.rnn_size))
end
for i = 1, opt.num_memory_modules do
    table.insert(inputs, torch.rand(opt.batch_size, n_memory_vectors, opt.memory_size))
    table.insert(inputs, torch.rand(opt.batch_size, n_memory_vectors))
    table.insert(inputs, torch.rand(opt.batch_size, opt.memory_size))
    table.insert(inputs, torch.rand(opt.batch_size, opt.memory_size))
end
print('\nForwards pass!')
print(unpack(rnn:forward(inputs)))

grads = {}
for i = 1, opt.num_lstm_layers do
    table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
    table.insert(grads, torch.rand(opt.batch_size, opt.rnn_size))
end
for i = 1, opt.num_memory_modules do
    table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 2, opt.memory_size))
    table.insert(grads, torch.rand(opt.batch_size, n_memory_vectors + 2))
    table.insert(grads, torch.rand(opt.batch_size, opt.memory_size))
    table.insert(grads, torch.rand(opt.batch_size, opt.memory_size))
end
table.insert(grads, torch.rand(opt.batch_size, opt.vocab_size))

print('\n\nBackwards pass!')
gradient = rnn:backward(inputs, grads)
print(unpack(gradient))
rnn:updateParameters(1)
print(#inputs)
print(#gradient)
