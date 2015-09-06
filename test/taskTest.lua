local Task = require '../task'

local tester = torch.Tester()
local tests = {}
local precision = 1e-6
local n_iters = 100

local opt = {
    min_train_seq_len = 5,
    max_train_seq_len = 10,
    min_test_seq_len = 15,
    max_test_seq_len = 20,
    batch_size = 2,
    vocab_size = 10
}

function tests.testGenerateSequence()
    for i = 1, n_iters do 
        local x, seq_len = Task.generateSequence(false, false, opt)
        tester:assertge(seq_len, opt.min_train_seq_len, 
            "seq_len less than min_train_seq_len!")
        tester:assertle(seq_len, opt.max_train_seq_len, 
            "seq_len greater than max_train_seq_len!")
        tester:assert(x:le(Task.end_token):sum() == 0, 
            "x has reserved fields in it!")
    end

    for i = 1, n_iters do 
        local x, seq_len = Task.generateSequence(false, true, opt)
        tester:assertge(seq_len, opt.min_test_seq_len, 
            "seq_len less than min_test_seq_len!")
        tester:assertle(seq_len, opt.max_test_seq_len, 
            "seq_len greater than max_test_seq_len!")
        tester:assert(x:le(Task.end_token):sum() == 0, 
            "x has reserved fields in it!")
    end

    for i = 1, n_iters do 
        local x, seq_len = Task.generateSequence(true, false, opt)
        tester:assert(seq_len % 2 == 0, "seq_len is not even!")
    end
end

function tests.testCopy()
    local x = torch.Tensor{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}:t()
    tester:assertTensorEq(torch.Tensor{1, 2, 3}, Task.copy(x, 1), precision,
        "Copy of index 1 is wrong!")
    tester:assertTensorEq(torch.Tensor{4, 5, 6}, Task.copy(x, 2), precision,
        "Copy of index 2 is wrong!")
    tester:assertTensorEq(torch.Tensor{7, 8, 9}, Task.copy(x, 3), precision,
        "Copy of index 3 is wrong!")
    tester:assertTensorEq(torch.Tensor{10, 11, 12}, Task.copy(x, 4), precision,
        "Copy of index 4 is wrong!")
end

function tests.testReverse()
    local x = torch.Tensor{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}:t()
    tester:assertTensorEq(torch.Tensor{10, 11, 12}, Task.reverse(x, 1, 4), precision,
        "Reverse of index 1 is wrong!")
    tester:assertTensorEq(torch.Tensor{7, 8, 9}, Task.reverse(x, 2, 4), precision,
        "Reverse of index 2 is wrong!")
    tester:assertTensorEq(torch.Tensor{4, 5, 6}, Task.reverse(x, 3, 4), precision,
        "Reverse of index 3 is wrong!")
    tester:assertTensorEq(torch.Tensor{1, 2, 3}, Task.reverse(x, 4, 4), precision,
        "Reverse of index 4 is wrong!")
end

function tests.testBigramFlip()
    local x = torch.Tensor{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}:t()
    tester:assertTensorEq(torch.Tensor{4, 5, 6}, Task.bigramFlip(x, 1), precision,
        "Bigram flip of index 1 is wrong!")
    tester:assertTensorEq(torch.Tensor{1, 2, 3}, Task.bigramFlip(x, 2), precision,
        "Bigram flip of index 2 is wrong!")
    tester:assertTensorEq(torch.Tensor{10, 11, 12}, Task.bigramFlip(x, 3), precision,
        "Bigram flip of index 3 is wrong!")
    tester:assertTensorEq(torch.Tensor{7, 8, 9}, Task.bigramFlip(x, 4), precision,
        "Bigram flip of index 4 is wrong!")
end

tester:add(tests)
tester:run()
