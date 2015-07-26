require 'nn'
require 'layers/CumulativeSum'

local tester = torch.Tester()
local tests = {}
local jac = nn.Jacobian
local precision = 1e-6

local cumSum = nn.CumulativeSum(false)
local reverseCumSum = nn.CumulativeSum(true)
local x = torch.Tensor{1, 2, 3, 4}

function tests.testForward()
    local xCumSum = torch.Tensor{1, 3, 6, 10}
    local xPredictedCumSum = cumSum:forward(x)
    tester:assertTensorEq(xCumSum, xPredictedCumSum, precision, 
                          "The cumulative sum is wrong!")
end

function tests.testReverseForward()
    local xReverseCumSum = torch.Tensor{10, 9, 7, 4}
    local xPredictedReverseCumSum = reverseCumSum:forward(x)
    tester:assertTensorEq(xReverseCumSum, xPredictedReverseCumSum, precision, 
                          "The reverse cumulative sum is wrong!")
end

function tests.testBackwards()
    local err = jac.testJacobian(cumSum, x)
    tester:assertlt(err, precision, "The backwards pass is wrong!")
end

function tests.testReverseBackwards()
    local err = jac.testJacobian(reverseCumSum, x)
    tester:assertlt(err, precision, "The reverse backwards pass is wrong!")
end

tester:add(tests)
tester:run()
