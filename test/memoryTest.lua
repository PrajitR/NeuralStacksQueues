require 'nn'
require 'nngraph'
local Memory = require 'Memory'

local tester = torch.Tester()
local tests = {}
local precision = 1e-6

function tests.testStackUpdateStrength()
    local prev_strength = nn.Identity()()
    local pop = nn.Identity()()
    local push = nn.Identity()()
    local new_strength = Memory.updateStrength(prev_strength, pop, push, true, true)
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
    local new_strength = Memory.updateStrength(prev_strength, pop, push, false, true)
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

function tests.testStackComputeRead()
    options = {memory_size = 3}
    local strength = nn.Identity()()
    local memory_vectors = nn.Identity()()
    local read = Memory.computeRead(strength, memory_vectors, true)
    local computeReadModule = nn.gModule({strength, memory_vectors}, {read})

    local mv = torch.Tensor{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}

    local s = torch.Tensor{{0.5, 0.4, 1.0}}:t()
    local expected = torch.Tensor{0.0, 0.0, 3.0}
    local predicted = computeReadModule:forward({s, mv})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackComputeRead fails with s={0.5, 0.4, 1.0}")

    s = torch.Tensor{{0.5, 0.4, 0.8}}:t()
    expected = torch.Tensor{0.0, 0.4, 2.4}
    predicted = computeReadModule:forward({s, mv})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackComputeRead fails with s={0.5, 0.4, 0.8}")

    s = torch.Tensor{{0.5, 0.4, 0.6}}:t()
    expected = torch.Tensor{0.0, 0.8, 1.8}
    predicted = computeReadModule:forward({s, mv})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackComputeRead fails with s={0.5, 0.4, 0.6}")

    s = torch.Tensor{{0.5, 0.3, 0.6}}:t()
    expected = torch.Tensor{0.1, 0.6, 1.8}
    predicted = computeReadModule:forward({s, mv})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackComputeRead fails with s={0.5, 0.3, 0.6}")

    s = torch.Tensor{{0.3, 0.3, 0.3}}:t()
    expected = torch.Tensor{0.3, 0.6, 0.9}
    predicted = computeReadModule:forward({s, mv})
    tester:assertTensorEq(expected, predicted, precision,
                          "testStackComputeRead fails with s={0.3, 0.6, 0.9}")
end

function tests.testQueueComputeRead()
    options = {memory_size = 3}
    local strength = nn.Identity()()
    local memory_vectors = nn.Identity()()
    local read = Memory.computeRead(strength, memory_vectors, false)
    local computeReadModule = nn.gModule({strength, memory_vectors}, {read})

    local V = torch.Tensor{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}

    local s = torch.Tensor{{1.0, 0.4, 0.5}}:t()
    local expected = torch.Tensor{1.0, 0.0, 0.0}
    local predicted = computeReadModule:forward({s, V})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueComputeRead fails with s={1.0, 0.4, 0.5}")

    s = torch.Tensor{{0.8, 0.3, 0.5}}:t()
    expected = torch.Tensor{0.8, 0.4, 0.0}
    predicted = computeReadModule:forward({s, V})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueComputeRead fails with s={0.8, 0.3, 0.5}")

    s = torch.Tensor{{0.6, 0.4, 0.5}}:t()
    expected = torch.Tensor{0.6, 0.8, 0.0}
    predicted = computeReadModule:forward({s, V})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueComputeRead fails with s={0.6, 0.4, 0.5}")

    s = torch.Tensor{{0.6, 0.3, 0.5}}:t()
    expected = torch.Tensor{0.6, 0.6, 0.3}
    predicted = computeReadModule:forward({s, V})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueComputeRead fails with s={0.6, 0.3, 0.5}")

    s = torch.Tensor{{0.3, 0.3, 0.3}}:t()
    expected = torch.Tensor{0.3, 0.6, 0.9}
    predicted = computeReadModule:forward({s, V})
    tester:assertTensorEq(expected, predicted, precision,
                          "testQueueComputeRead fails with s={0.3, 0.6, 0.9}")
end

function tests.testStack()
    options = {memory_size = 3}
    local stack = Memory.Stack()
    
    local V = torch.Tensor{{0.0, 0.0, 0.0}}:t()
    local s = torch.Tensor{{0.0}}
    local v = torch.Tensor{{1.0, 0.0, 0.0}}:t()
    local u = torch.Tensor{0.0}
    local d = torch.Tensor{{0.8}}

    local predicted_V, predicted_s, predicted_read = 
        unpack(stack:forward({V, s, v, u, d}))
    local expected_V = torch.Tensor{{0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}}
    local expected_s = torch.Tensor{{0.0, 0.8}}:t()
    local expected_read = torch.Tensor{0.8, 0.0, 0.0}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testStack wrong V in first update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testStack wrong s in first update")
    tester:assertTensorEq(expected_read, predicted_read, precision,
                         "testStack wrong read in first update")

    V:resizeAs(predicted_V):copy(predicted_V)
    s:resizeAs(predicted_s):copy(predicted_s)
    v = torch.Tensor{{0.0, 2.0, 0.0}}:t()
    u = torch.Tensor{0.1}
    d = torch.Tensor{{0.5}}

    predicted_V, predicted_s, predicted_read = 
        unpack(stack:forward({V, s, v, u, d}))
    expected_V = torch.Tensor{{0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 0.0}}
    expected_s = torch.Tensor{{0.0, 0.7, 0.5}}:t()
    expected_read = torch.Tensor{0.5, 1.0, 0.0}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testStack wrong V in second update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testStack wrong s in second update")
    tester:assertTensorEq(expected_read, predicted_read, precision,
                         "testStack wrong read in second update")

    V:resizeAs(predicted_V):copy(predicted_V)
    s:resizeAs(predicted_s):copy(predicted_s)
    v = torch.Tensor{{0.0, 0.0, 3.0}}:t()
    u = torch.Tensor{0.9}
    d = torch.Tensor{{0.9}}

    predicted_V, predicted_s, predicted_read = 
        unpack(stack:forward({V, s, v, u, d}))
    expected_V = torch.Tensor{
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 2.0, 0.0},
        {0.0, 0.0, 0.0, 3.0}}
    expected_s = torch.Tensor{{0.0, 0.3, 0.0, 0.9}}
    expected_read = torch.Tensor{0.1, 0.0, 2.7}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testStack wrong V in third update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testStack wrong s in third update")
    tester:assertTensorEq(expected_read, predicted_read, precision,
                         "testStack wrong read in third update")
end

function tests.testQueue()
    options = {memory_size = 3}
    local queue = Memory.Queue()
    
    local V = torch.Tensor{{0.0, 0.0, 0.0}}:t()
    local s = torch.Tensor{{0.0}}
    local v = torch.Tensor{{1.0, 0.0, 0.0}}:t()
    local u = torch.Tensor{0.0}
    local d = torch.Tensor{{0.8}}

    local predicted_V, predicted_s, predicted_read = 
        unpack(queue:forward({V, s, v, u, d}))
    local expected_V = torch.Tensor{{0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}}
    local expected_s = torch.Tensor{{0.0, 0.8}}:t()
    local expected_read = torch.Tensor{0.8, 0.0, 0.0}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testQueue wrong V in first update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testQueue wrong s in first update")
    tester:assertTensorEq(expected_read, predicted_read, precision,
                         "testQueue wrong read in first update")

    V:resizeAs(predicted_V):copy(predicted_V)
    s:resizeAs(predicted_s):copy(predicted_s)
    v = torch.Tensor{{0.0, 2.0, 0.0}}:t()
    u = torch.Tensor{0.1}
    d = torch.Tensor{{0.5}}

    predicted_V, predicted_s, predicted_read = 
        unpack(queue:forward({V, s, v, u, d}))
    expected_V = torch.Tensor{{0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 0.0}}
    expected_s = torch.Tensor{{0.0, 0.7, 0.5}}:t()
    expected_read = torch.Tensor{0.7, 0.6, 0.0}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testQueue wrong V in second update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testQueue wrong s in second update")
    tester:assertTensorEq(expected_read, predicted_read, precision,
                         "testQueue wrong read in second update")

    V:resizeAs(predicted_V):copy(predicted_V)
    s:resizeAs(predicted_s):copy(predicted_s)
    v = torch.Tensor{{0.0, 0.0, 3.0}}:t()
    u = torch.Tensor{0.8}
    d = torch.Tensor{{0.9}}

    predicted_V, predicted_s, predicted_read = 
        unpack(queue:forward({V, s, v, u, d}))
    expected_V = torch.Tensor{
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 2.0, 0.0},
        {0.0, 0.0, 0.0, 3.0}}
    expected_s = torch.Tensor{{0.0, 0.0, 0.4, 0.9}}
    expected_read = torch.Tensor{0.0, 0.8, 1.8}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testQueue wrong V in third update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testQueue wrong s in third update")
    tester:assertTensorEq(expected_read, predicted_read, precision,
                         "testQueue wrong read in third update")
end

function tests.testDeQue()
--function testDeQue()
    options = {memory_size = 3}    
    local deque = Memory.DeQue()

    local V = torch.Tensor{{0.0, 0.0, 0.0}}:t()
    local s = torch.Tensor{{0.0}}
    local vt = torch.Tensor{{1.0, 0.0, 0.0}}:t()
    local vb = torch.Tensor{{0.0, 1.0, 0.0}}:t()
    local ut = torch.Tensor{0.0}
    local ub = torch.Tensor{0.0}
    local dt = torch.Tensor{{0.9}}
    local db = torch.Tensor{{0.2}}

    local predicted_V, predicted_s, predicted_read_top, predicted_read_bot = 
        unpack(deque:forward({V, s, vt, vb, ut, ub, dt, db}))
    local expected_V = torch.Tensor{
        {0.0, 0.0, 1.0}, 
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}}
    local expected_s = torch.Tensor{{0.2, 0.0, 0.9}}:t()
    local expected_read_top = torch.Tensor{0.9, 0.1, 0.0}
    local expected_read_bot = torch.Tensor{0.8, 0.2, 0.0}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testDeQue wrong V in first update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testDeQue wrong s in first update")
    tester:assertTensorEq(expected_read_top, predicted_read_top, precision,
                         "testDeQue wrong read top in first update")
    tester:assertTensorEq(expected_read_bot, predicted_read_bot, precision,
                         "testDeQue wrong read bot in first update")

    V:resizeAs(predicted_V):copy(predicted_V)
    s:resizeAs(predicted_s):copy(predicted_s)
    vt = torch.Tensor{{0.0, 0.0, 1.0}}:t()
    vb = torch.Tensor{{0.0, 0.5, 0.5}}:t()
    ut = torch.Tensor{0.6}
    ub = torch.Tensor{0.3}
    dt = torch.Tensor{{0.7}}
    db = torch.Tensor{{0.5}}

    predicted_V, predicted_s, predicted_read_top, predicted_read_bot = 
        unpack(deque:forward({V, s, vt, vb, ut, ub, dt, db}))
    expected_V = torch.Tensor{
        {0.0, 0.0, 0.0, 1.0, 0.0},
        {0.5, 1.0, 0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0, 0.0, 1.0}}
    expected_s = torch.Tensor{0.5, 0.0, 0.0, 0.2, 0.7}
    expected_read_top = torch.Tensor{0.2, 0.05, 0.75}
    expected_read_bot = torch.Tensor{0.2, 0.25, 0.55}

    tester:assertTensorEq(expected_V, predicted_V, precision,
                         "testDeQue wrong V in second update")
    tester:assertTensorEq(expected_s, predicted_s, precision,
                         "testDeQue wrong s in second update")
    tester:assertTensorEq(expected_read_top, predicted_read_top, precision,
                         "testDeQue wrong read top in second update")
    tester:assertTensorEq(expected_read_bot, predicted_read_bot, precision,
                         "testDeQue wrong read bot in second update")
end

tester:add(tests)
tester:run()
