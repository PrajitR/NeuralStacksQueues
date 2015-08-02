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
    -- for DeQue, can only append after both passes over the vector
    local should_append = push ~= nil
    local new_strength = nil
    if should_append then 
        new_strength = nn.JoinTable(2)({outer_max, push})
    else
        new_strength = outer_max
    end
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

function Memory.twoSidedMemory()
    local prev_memory_vectors = nn.Identity()()
    local prev_strength = nn.Identity()()
    local memory_top = nn.Identity()()
    local memory_bot = nn.Identity()()
    local pop_top = nn.Identity()()
    local pop_bot = nn.Identity()()
    local push_top = nn.Identity()()
    local push_bot = nn.Identity()()

    local new_memory_vectors = nn.JoinTable(2)(
        {memory_bot, prev_memory_vectors, memory_top})
    local strength_top = Memory.updateStrength(prev_strength, pop_top, nil, true)
    local strength_both = Memory.updateStrength(strength_top, pop_bot, nil, false)
    local new_strength = nn.JoinTable(1)({push_bot, strength_both, push_top})
    local read_top = Memory.computeRead(new_strength, new_memory_vectors, true)
    local read_bot = Memory.computeRead(new_strength, new_memory_vectors, false)

    return nn.gModule(
        {prev_memory_vectors, prev_strength, memory_top, memory_bot,
            pop_top, pop_bot, push_top, push_bot},
        {new_memory_vectors, new_strength, read_top, read_bot})
        --{new_memory_vectors, strength_both, push_top, push_bot})
end

function Memory.Stack()
    return Memory.oneSidedMemory(true)
end

function Memory.Queue()
    return Memory.oneSidedMemory(false)
end

function Memory.DeQue()
    return Memory.twoSidedMemory()
end

return Memory
