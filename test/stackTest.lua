require 'nn'
require 'nngraph'
local Stack = require 'stack'

local tester = torch.Tester()
local tests = {}
local precision = 1e-6

function tests.testStackUpdateStrength()
    local prev_strength = nn.Identity()()
    local pop = nn.Identity()()
    local push = nn.Identity()()
    local new_strength = Stack.updateStrength(prev_strength, pop, push, true)
    local updateStrengthModule = nn.gModule({prev_strength, pop, push}, {new_strength})

    local s = torch.Tensor{{0.4, 0.1, 0.3}}:t()
    local d = torch.Tensor{0.6}

    local u = torch.Tensor{0.20}
    local expected = torch.Tensor{0.4, 0.1, 0.1, 0.6}
    local predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackUpdateStrength fails with u=0.25")

    u = torch.Tensor{0.30}
    expected = torch.Tensor{0.4, 0.1, 0.0, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackUpdateStrength fails with u=0.30")

    u = torch.Tensor{0.35}
    expected = torch.Tensor{0.4, 0.05, 0.0, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackUpdateStrength fails with u=0.35")

    u = torch.Tensor{0.40}
    expected = torch.Tensor{0.4, 0.0, 0.0, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackUpdateStrength fails with u=0.40")

    u = torch.Tensor{0.50}
    expected = torch.Tensor{0.3, 0.0, 0.0, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackUpdateStrength fails with u=0.50")

    u = torch.Tensor{0.90}
    expected = torch.Tensor{0.0, 0.0, 0.0, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackUpdateStrength fails with u=0.90")
end

function tests.testQueueUpdateStrength()
    local prev_strength = nn.Identity()()
    local pop = nn.Identity()()
    local push = nn.Identity()()
    local new_strength = Stack.updateStrength(prev_strength, pop, push, false)
    local updateStrengthModule = nn.gModule({prev_strength, pop, push}, {new_strength})

    local s = torch.Tensor{{0.4, 0.1, 0.3}}:t()
    local d = torch.Tensor{0.6}

    local u = torch.Tensor{0.20}
    local expected = torch.Tensor{0.2, 0.1, 0.3, 0.6}
    local predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueUpdateStrength fails with u=0.20")

    u = torch.Tensor{0.40}
    expected = torch.Tensor{0.0, 0.1, 0.3, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueUpdateStrength fails with u=0.40")

    u = torch.Tensor{0.45}
    expected = torch.Tensor{0.0, 0.05, 0.3, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueUpdateStrength fails with u=0.45")

    u = torch.Tensor{0.50}
    expected = torch.Tensor{0.0, 0.0, 0.3, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueUpdateStrength fails with u=0.50")

    u = torch.Tensor{0.60}
    expected = torch.Tensor{0.0, 0.0, 0.2, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueUpdateStrength fails with u=0.60")

    u = torch.Tensor{0.90}
    expected = torch.Tensor{0.0, 0.0, 0.0, 0.6}
    predicted = updateStrengthModule:forward({s, u, d})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueUpdateStrength fails with u=0.90")
end

tester:add(tests)
tester:run()
