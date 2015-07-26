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

return Stack
