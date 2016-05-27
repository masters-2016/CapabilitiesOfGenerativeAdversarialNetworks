require 'torch'
require 'optim'

function default_options()
    return {
        learningRate = 0.001,
        beta1 = 0.5,
        epochs = 25,
		batchSize = 64,
		inputDim = {3, 64, 64},
		outputDim = {3, 64, 64},
        gpu = false
    }
end

-- This function is from the DCGAN code
function cudnn(net)
    for k, l in ipairs(net.modules) do
        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
						 l.kW, l.kH, l.dW, l.dH, 
						 l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
    return net
end

function train(net, x, y, options, callback)
    print(options)

    local criterion = nn.BCECriterion()

    optimState = {
        learningRate = options.learningRate,
        beta1 = options.beta1,
    }

    local input = torch.Tensor(options.batchSize, table.unpack(options.inputDim))
    local target = torch.Tensor(options.batchSize, table.unpack(options.outputDim))

    local err = 0.0

    if options.gpu then
        require 'cunn'
        cutorch.setDevice(options.gpu)

        input = input:cuda()
        target = target:cuda()

        net = cudnn(net)
        net:cuda()

        criterion:cuda()
    end

    local total_timer = torch.Timer()
    local epoch_timer = torch.Timer()

    local parameters, gradParameters = net:getParameters()

    local optimFunc = function(x)
        gradParameters:zero()

        local output = net:forward(input)
        err = criterion:forward(output, target)
        net:backward(input, criterion:backward(output, target))

        return err, gradParameters
    end

    for e = 1, options.epochs do
        epoch_timer:reset()

        for i = 1, options.batchSize do
            local idx = math.random(1, x:size(1))
            input[i] = x[idx]
            target[i] = y[idx]
        end

        optim.adam(optimFunc, parameters, optimState)

        print(('epoch %d,epoch_time %.3f,total_time %.3f,err %.6f'):format(e, epoch_timer:time().real, total_timer:time().real, err))

        if callback then
            callback(e)
        end
    end
end
