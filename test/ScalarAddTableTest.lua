require 'nn'
require 'layers/ScalarAddTable'

local tester = torch.Tester()
local tests = {}
local precision = 1e-6

local add = nn.ScalarAddTable()

function tests.testForward()
    local x = torch.Tensor{{1, 2, 3}, {4, 5, 6}}
    local y = torch.Tensor{{5}, {-7}}

    local expected = torch.Tensor{{6, 7, 8}, {-3, -2, -1}}
    local predicted = add:forward({x, y})
    tester:assertTensorEq(expected, predicted, precision,
                          "The forward pass is wrong!")
end

function tests.testBackwards() 
    local x = torch.rand(1, 5)
    local y = torch.rand(1, 1)
    local eps = 1e-6
    local z = add:forward({x, y}):clone()
    local jac = torch.DoubleTensor(z:size(2), x:size(2) + 1)
    
    local one_hot = torch.zeros(1, z:size(2))
    for i = 1, z:size(2) do
        one_hot[{1, i}] = 1
        local back = add:backward({x, y}, one_hot)
        jac[i] = torch.cat(back[1], back[2])
        one_hot[{1, i}] = 0
    end

    local jac_est = torch.DoubleTensor(z:size(2), x:size(2) + 1)
    for i = 1, x:size(2) do
        x[{1, i}] = x[{1, i}] + eps
        jac_est[{{}, i}]:copy(add:forward({x, y}))
        x[{1, i}] = x[{1, i}] - 2 * eps
        jac_est[{{}, i}]:add(-1, add:forward({x, y})):div(2 * eps)
        x[{1, i}] = x[{1, i}] + eps
    end
    
    y[{1, 1}] = y[{1, 1}] + eps
    jac_est[{{}, -1}]:copy(add:forward({x, y}))
    y[{1, 1}] = y[{1, 1}] - 2 * eps
    jac_est[{{}, -1}]:add(-1, add:forward({x, y})):div(2 * eps)
    y[{1, 1}] = y[{1, 1}] + eps

    tester:assertTensorEq(jac, jac_est, precision, 
                          "The backwards pass is wrong!")
end

tester:add(tests)
tester:run()
