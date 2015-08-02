local CumulativeSum, parent = torch.class('nn.CumulativeSum', 'nn.Module')

function CumulativeSum:__init(reverse)
    parent.__init(self)
    self.reverse = reverse
    self.input_sum = torch.Tensor()
end

function CumulativeSum:reverseCumsum(internal, input)
    internal:resizeAs(input):cumsum(input, 2)
    self.input_sum:resizeAs(input)
    self.input_sum:set(torch.sum(input, 2):repeatTensor(1, input:size(2)))
    internal:mul(-1):add(self.input_sum):add(input)
    return cumsum
end

function CumulativeSum:updateOutput(input)
    if not self.reverse then
       self.output:resizeAs(input):cumsum(input, 2)
    else
       self:reverseCumsum(self.output, input)
    end
    return self.output
end

function CumulativeSum:updateGradInput(input, gradOutput)
    if self.reverse then
        self.gradInput:resizeAs(gradOutput):cumsum(gradOutput, 2)
    else
        self:reverseCumsum(self.gradInput, gradOutput)
    end
    return self.gradInput
end
