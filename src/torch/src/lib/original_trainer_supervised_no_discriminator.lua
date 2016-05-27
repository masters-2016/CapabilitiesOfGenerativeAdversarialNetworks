require 'torch'
require 'nn'
require 'optim'

util = paths.dofile('../lib/util.lua')

function train_original_supervised_no_discriminator(netG, netD, data, opt, callback, netC)
    local criterion = nn.BCECriterion()

    optimStateG = {
        learningRate = opt.lr,
        beta1 = opt.beta1,
    }
    optimStateC = {
        learningRate = opt.lr,
        beta1 = opt.beta1,
    }

    local input = torch.Tensor(opt.batchSize, unpack(opt.n))

    local noise_count = opt.nz
    noise_count[1] = noise_count[1] + opt.labelCount
    local noise = torch.Tensor(opt.batchSize, unpack(noise_count))

    local classes = torch.Tensor(opt.batchSize, opt.labelCount)

    local errG, errC
    local epoch_tm = torch.Timer()
    local tm = torch.Timer()
    local data_tm = torch.Timer()

    if opt.gpu > 0 then
        require 'cunn'
        cutorch.setDevice(opt.gpu)

        input = input:cuda()
        noise = noise:cuda()
        classes = classes:cuda()

        netG = util.cudnn(netG)
        netC = util.cudnn(netC)

        netG:cuda()
        netC:cuda()
        criterion:cuda()
    end

    local parametersG, gradParametersG = netG:getParameters()
    local parametersC, gradParametersC = netC:getParameters()

    -- Define a function to generate random noise and labels
    local generateNoise = function()
        if opt.noise == 'uniform' then -- regenerate random noise
            noise:uniform(-1, 1)
        elseif opt.noise == 'normal' then
            noise:normal(0, 1)
        end

        for i = 1, opt.batchSize do
            for j = 1, opt.labelCount do
                noise[i][j][1][1] = math.random(0, 1) * 2 - 1
            end
        end
    end

    -- create closure to evaluate f(X) and df/dX of classifier
    local fCx = function(x)
        netC:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

        gradParametersC:zero()

        -- train with real
        data_tm:stop()
        local batch = data.random_batch(opt.batchSize)
        input:copy(batch.data)
        classes:copy(batch.labels)

        local output = netC:forward(input)
        local errC_real = criterion:forward(output, classes)
        local df_do = criterion:backward(output, classes)
        netC:backward(input, df_do)

        -- train with fake
        generateNoise()
        local fake = netG:forward(noise)
        input:copy(fake)
        classes:fill(0)

        local output = netC:forward(input)
        local errC_fake = criterion:forward(output, classes)
        local df_do = criterion:backward(output, classes)
        netC:backward(input, df_do)

        errC = errC_real + errC_fake

        return errC, gradParametersC
    end

    -- create closure to evaluate f(X) and df/dX of generator
    local fGx = function(x)
        netC:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        
        gradParametersG:zero()

        -- Train against C
        generateNoise()
        local fake = netG:forward(noise)
        input:copy(fake)

        for i = 1, opt.batchSize do
            for j = 1, opt.labelCount do
                classes[i][j] = noise[i][j]
            end
        end

        local output = netC:forward(input)

        errG = criterion:forward(output, classes)
        local df_do = criterion:backward(output, classes)
        local df_dg = netC:updateGradInput(input, df_do)

        netG:backward(noise, df_dg)

        return errG, gradParametersG
    end

    -- Training functions
    local function trainC()
        parametersC, gradParametersC = nil, nil -- nil them to avoid spiking memory
        parametersC, gradParametersC = netC:getParameters() -- reflatten the params and get them

        optim.adam(fCx, parametersC, optimStateC)
    end

    local function trainG()
        parametersG, gradParametersG = nil, nil -- nil them to avoid spiking memory
        parametersG, gradParametersG = netG:getParameters() -- reflatten the params and get them

        optim.adam(fGx, parametersG, optimStateG)
    end

    -- train
    local epoch = 0
    while opt.maxEpochs and epoch < opt.maxEpochs do
        epoch = epoch + 1

        epoch_tm:reset()
        local counter = 0

        tm:reset()

        for x = 1, opt.k do
            trainC()
        end

        trainG()

        if not callback(epoch, epoch_tm:time().real, 0.0, errG, errC) then
            break
        end
    end
end
