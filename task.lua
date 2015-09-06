local Task = {}
Task.start_token_val = 1
Task.sep_token_val = 2
Task.end_token_val = 3

function Task.generateSequence(make_even, is_test, opt)
    local min_len = is_test and opt.min_test_seq_len or opt.min_train_seq_len 
    local max_len = is_test and opt.max_test_seq_len or opt.max_train_seq_len 
    --local seq_len = torch.floor(torch.uniform(min_len, max_len + 1))
    local seq_len = 2
    if make_even and seq_len % 2 == 1 then
        seq_len = seq_len + 1
    end

    local x = torch.rand(opt.batch_size, seq_len)
                    :mul(opt.vocab_size - Task.end_token_val)
                    :floor()
                    :add(Task.end_token_val + 1)
    return x, seq_len
end

function Task.copy(x, index)
    return x[{{}, index}]
end

function Task.reverse(x, index)
    local seq_len = x:size(2)
    return x[{{}, seq_len - index + 1}]
end

function Task.bigramFlip(x, index)
    if index % 2 == 1 then
        return x[{{}, index + 1}]
    else
        return x[{{}, index - 1}]
    end
end

function Task.startToken(opt)
    if Task.start_token_tensor == nil then
        Task.start_token_tensor 
            = torch.Tensor(opt.batch_size):fill(Task.start_token_val)
    end
    return Task.start_token_tensor
end

function Task.sepToken(opt)
    if Task.sep_token_tensor == nil then
        Task.sep_token_tensor = 
            torch.Tensor(opt.batch_size):fill(Task.sep_token_val)
    end
    return Task.sep_token_tensor
end

function Task.endToken(opt)
    if Task.end_token_tensor == nil then
        Task.end_token_tensor 
            = torch.Tensor(opt.batch_size):fill(Task.end_token_val)
    end
    return Task.end_token_tensor
end

return Task
