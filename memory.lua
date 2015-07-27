require 'nn'
require 'nngraph'
require 'layers/CumulativeSum'
require 'layers/ScalarAddTable'

local Memory = {}

function Memory.updateStrength(prev_strength, pop, push, is_stack)
    local neg_cumsum = nn.MulConstant(-1)(nn.CSubTable()(
                        { nn.CumulativeSum(is_stack)(prev_strength), 
                          prev_strength }))
    local inner_max = nn.ReLU()(nn.ScalarAddTable()({neg_cumsum, pop}))
    local outer_max = nn.ReLU()(nn.CSubTable()({prev_strength, inner_max}))
    local new_strength = nn.JoinTable(1)({outer_max, push})
    return new_strength
end

function Memory.computeRead(strength, memory_vectors, is_stack)
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

function Memory.oneSidedMemory(is_stack)
    local prev_memory_vectors = nn.Identity()()
    local prev_strength = nn.Identity()()
    local new_memory = nn.Identity()()
    local pop = nn.Identity()()
    local push = nn.Identity()()

    local new_memory_vectors = nn.JoinTable(2)({prev_memory_vectors, new_memory})
    local new_strength = Memory.updateStrength(prev_strength, pop, push, is_stack)
    local read = Memory.computeRead(new_strength, new_memory_vectors, is_stack)
    
    return nn.gModule(
        {prev_memory_vectors, prev_strength, new_memory, pop, push},
        {new_memory_vectors, new_strength, read})
end

function Memory.Stack()
    return Memory.oneSidedMemory(true)
end

function Memory.Queue()
    return Memory.oneSidedMemory(false)
end

return Memory
