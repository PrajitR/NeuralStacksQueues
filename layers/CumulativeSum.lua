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
    if not reverse then
       return torch.cumsum(input)
    else
        return reverseCumsum(input)
    end
end

function CumulativeSum:updateGradInput(input, gradOutput)
    if reverse then
        return torch.cumsum(gradOutput)
    else
        return reverseCumsum(gradOutput)
    end
end
