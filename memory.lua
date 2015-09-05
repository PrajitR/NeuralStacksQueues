require 'nn'
require 'nngraph'
require 'layers/CumulativeSum'
require 'layers/ScalarAddTable'

local Memory = {}

function Memory.updateStrength(prev_strength, pop, is_stack)
    local neg_cumsum = nn.MulConstant(-1)(nn.CSubTable()(
                        { nn.CumulativeSum(is_stack)(prev_strength), 
                          prev_strength }))
    local inner_max = nn.ReLU()(nn.ScalarAddTable()({neg_cumsum, pop}))
    local outer_max = nn.ReLU()(nn.CSubTable()({prev_strength, inner_max}))
    return outer_max
end

function Memory.computeRead(strength, memory_vectors, is_stack, opt)
    local neg_cumsum = nn.MulConstant(-1)(nn.CSubTable()(
                        { nn.CumulativeSum(is_stack)(strength), 
                          strength }))
    local inner_max = nn.ReLU()(nn.AddConstant(1)(neg_cumsum))
    local coeff = nn.Min(1)(nn.JoinTable(1)(
                            { nn.View(1, opt.batch_size, -1)(strength), 
                              nn.View(1, opt.batch_size, -1)(inner_max) }))
    local read = nn.MixtureTable(2)({coeff, memory_vectors})
    return read
end

function Memory.oneSidedMemory(prev_memory_vectors, prev_strength,
                               new_memory, pop, push, is_stack, opt)

    local new_memory_vectors = nn.JoinTable(2)(
        { prev_memory_vectors, 
          nn.Reshape(1, opt.memory_size, true)(new_memory) })
    local updated_strength = Memory.updateStrength(prev_strength, pop, is_stack)
    local new_strength = nn.JoinTable(2)({updated_strength, push})
    local read = Memory.computeRead(new_strength, new_memory_vectors, is_stack, opt)
    return new_memory_vectors, new_strength, read
end

function Memory.Stack(prev_memory_vectors, prev_strength,
                      new_memory, pop, push, opt)
    return Memory.oneSidedMemory(prev_memory_vectors, prev_strength, 
                new_memory, pop, push, true, opt)
end

function Memory.Queue(prev_memory_vectors, prev_strength,
                      new_memory, pop, push, opt)
    return Memory.oneSidedMemory(prev_memory_vectors, prev_strength, 
                new_memory, pop, push, false, opt)
end

function Memory.DeQue(prev_memory_vectors, prev_strength,
                      memory_top, memory_bot, pop_top, pop_bot,
                      push_top, push_bot, opt)

    local new_memory_vectors = nn.JoinTable(2)(
        { nn.Reshape(1, opt.memory_size, true)(memory_bot),
          prev_memory_vectors,
          nn.Reshape(1, opt.memory_size, true)(memory_top) })
    local strength_top = Memory.updateStrength(prev_strength, pop_top, true)
    local strength_both = Memory.updateStrength(strength_top, pop_bot, false)
    local new_strength = nn.JoinTable(2)({push_bot, strength_both, push_top})
    local read_top = Memory.computeRead(new_strength, new_memory_vectors, true, opt)
    local read_bot = Memory.computeRead(new_strength, new_memory_vectors, false, opt)

    return new_memory_vectors, new_strength, read_top, read_bot
    end

function Memory.oneSidedMemoryModule(MemoryType, opt)
    local prev_memory_vectors = nn.Identity()()
    local prev_strength = nn.Identity()()
    local new_memory = nn.Identity()()
    local pop = nn.Identity()()
    local push = nn.Identity()()

    local new_memory_vectors, new_strength, read = 
        MemoryType(prev_memory_vectors, prev_strength, new_memory, pop, push, opt)

    return nn.gModule(
        {prev_memory_vectors, prev_strength, new_memory, pop, push},
        {new_memory_vectors, new_strength, read})
end

function Memory.StackModule(opt)
    return Memory.oneSidedMemoryModule(Memory.Stack, opt)
end

function Memory.QueueModule(opt)
    return Memory.oneSidedMemoryModule(Memory.Queue, opt)
end

function Memory.DeQueModule(opt)
    local prev_memory_vectors = nn.Identity()()
    local prev_strength = nn.Identity()()
    local memory_top = nn.Identity()()
    local memory_bot = nn.Identity()()
    local pop_top = nn.Identity()()
    local pop_bot = nn.Identity()()
    local push_top = nn.Identity()()
    local push_bot = nn.Identity()()

    local new_memory_vectors, new_strength, read_top, read_bot =
        Memory.DeQue(prev_memory_vectors, prev_strength, memory_top,
                     memory_bot, pop_top, pop_bot, push_top, push_bot, opt)

    return nn.gModule(
        {prev_memory_vectors, prev_strength, memory_top, memory_bot,
            pop_top, pop_bot, push_top, push_bot},
        {new_memory_vectors, new_strength, read_top, read_bot})
end

return Memory
