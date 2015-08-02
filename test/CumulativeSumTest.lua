require 'nn'
require 'layers/CumulativeSum'

local tester = torch.Tester()
local tests = {}
local jac = nn.Jacobian
local precision = 1e-6

local cumsum = nn.CumulativeSum(false)
local reverse_cumsum = nn.CumulativeSum(true)
local x = torch.Tensor{{1, 2, 3, 4}, {5, 6, 7, 8}}

function tests.testForward()
    local expected = torch.Tensor{{1, 3, 6, 10}, {5, 11, 18, 26}}
    local predicted = cumsum:forward(x)
    tester:assertTensorEq(expected, predicted, precision, 
                          "The cumulative sum is wrong!")
end

function tests.testReverseForward()
    local expected = torch.Tensor{{10, 9, 7, 4}, {26, 21, 15, 8}}
    local predicted = reverse_cumsum:forward(x)
    tester:assertTensorEq(expected, predicted, precision, 
                          "The reverse cumulative sum is wrong!")
end

function tests.testBackwards()
    local err = jac.testJacobian(cumsum, torch.rand(4, 2))
    tester:assertlt(err, precision, "The backwards pass is wrong!")
    err = jac.testJacobian(cumsum, torch.rand(4, 8))
    tester:assertlt(err, precision, "The backwards pass is wrong!")
end

function tests.testReverseBackwards()
    local err = jac.testJacobian(reverse_cumsum, torch.rand(4, 2))
    tester:assertlt(err, precision, "The reverse backwards pass is wrong!")
    err = jac.testJacobian(reverse_cumsum, torch.rand(4, 8))
    tester:assertlt(err, precision, "The reverse backwards pass is wrong!")
end

tester:add(tests)
tester:run()
