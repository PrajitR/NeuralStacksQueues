local Task = {}
Task.start_token = 1
Task.sep_token = 2
Task.end_token = 3

function Task.generateSequence(even, test, opt)
    local min_len = test and opt.min_test_seq_len or opt.min_train_seq_len 
    local max_len = test and opt.max_test_seq_len or opt.max_train_seq_len 
    local seq_len = torch.floor(torch.uniform(min_len, max_len + 1))
    if even and seq_len % 2 == 1 then
        seq_len = seq_len + 1
    end

    local x = torch.rand(opt.batch_size, seq_len)
                :mul(opt.vocab_size - Task.end_token):floor():add(Task.end_token + 1)
    return x, seq_len
end

function Task.copy(x, index)
    return x[{{}, index}]
end

function Task.reverse(x, index, seq_len)
    return x[{{}, seq_len - index + 1}]
end

function Task.bigramFlip(x, index)
    if index % 2 == 1 then
        return x[{{}, index + 1}]
    else
        return x[{{}, index - 1}]
    end
end

return Task
