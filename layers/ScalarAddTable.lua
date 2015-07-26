local ScalarAddTable, parent = torch.class('nn.ScalarAddTable', 'nn.Module')

function ScalarAddTable:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function ScalarAddTable:updateOutput(input)
    local vector, scalar = unpack(input)
    self.output:set(vector + scalar[1])
    return self.output
end

function ScalarAddTable:updateGradInput(input, gradOutput)
    local vector, scalar = unpack(input)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(scalar)

    self.gradInput[1]:set(gradOutput)
    self.gradInput[2][1] = torch.sum(gradOutput)
    return self.gradInput
end
