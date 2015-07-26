local CumulativeSum, parent = torch.class('nn.CumulativeSum', 'nn.Module')

function CumulativeSum:__init(reverse)
    parent.__init(self)
    self.reverse = reverse
end

function reverseCumsum(input)
    local cumsum = torch.cumsum(input)
    cumsum:mul(-1):add(torch.sum(input)):add(input)
    return cumsum
end

function CumulativeSum:updateOutput(input)
    if not self.reverse then
       self.output = torch.cumsum(input)
    else
       self.output = reverseCumsum(input)
    end
    return self.output
end

function CumulativeSum:updateGradInput(input, gradOutput)
    if self.reverse then
        self.gradInput = torch.cumsum(gradOutput)
    else
        self.gradInput =  reverseCumsum(gradOutput)
    end
    return self.gradInput
end
