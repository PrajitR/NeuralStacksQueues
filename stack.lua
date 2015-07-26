require 'nn'
require 'nngraph'
require 'layers/CumulativeSum'
require 'layers/ScalarAddTable'

local Stack = {}

function Stack.updateStrength(prev_strength, pop, push, is_stack)
    local neg_cumsum = nn.MulConstant(-1)(nn.CSubTable()(
                        { nn.CumulativeSum(is_stack)(prev_strength), 
                          prev_strength }))
    local inner_max = nn.ReLU()(nn.ScalarAddTable()({neg_cumsum, pop}))
    local outer_max = nn.ReLU()(nn.CSubTable()({prev_strength, inner_max}))
    local new_strength = nn.JoinTable(1)({outer_max, push})
    return new_strength
end

function Stack.computeRead(strength, memory_vectors, is_stack)
    local neg_cumsum = nn.MulConstant(-1)(nn.CSubTable()(
                        { nn.CumulativeSum(is_stack)(strength), 
                          strength }))
    local inner_max = nn.ReLU()(nn.AddConstant(1)(neg_cumsum))
    local coeff = nn.Min(2)(nn.JoinTable(2)({strength, inner_max}))
    local scaled_memory = nn.CMulTable()({ 
        memory_vectors, 
        nn.Replicate(options.memory_size)(coeff)})
    local read = nn.Sum(2)(scaled_memory)
    return read
end

return Stack
