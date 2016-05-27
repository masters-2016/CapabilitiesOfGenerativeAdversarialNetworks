require 'torch'
require 'nn'
require 'optim'
require 'nngraph'

util = paths.dofile('../lib/util.lua')

function train_original_conditional(netG, netD, data, opt, callback)

    local real_label = 1
    local fake_label = 0

    local criterion = nn.BCECriterion()

    optimStateG = {
        learningRate = opt.lr,
        beta1 = opt.beta1,
    }
    optimStateD = {
        learningRate = opt.lr,
        beta1 = opt.beta1,
    }

    local input = torch.Tensor(opt.batchSize, unpack(opt.n))

    -- The input is both noise and parameters / labels
    local noise_count = opt.nz
    noise_count[1] = noise_count[1] + opt.labelCount
    local noise = torch.Tensor(opt.batchSize, unpack(noise_count))

    -- Output is a probability plus a number of labels
    local label = torch.Tensor(opt.batchSize)
    local classes = torch.Tensor(opt.batchSize, opt.labelCount)

    local errD, errG
    local epoch_tm = torch.Timer()
    local tm = torch.Timer()

    if opt.gpu > 0 then
        require 'cunn'
        cutorch.setDevice(opt.gpu)

        input = input:cuda()
        noise = noise:cuda()
        label = label:cuda()
        classes = classes:cuda()

        netG = util.cudnn(netG)
        netD = util.cudnn(netD)

        netD:cuda()
        netG:cuda()
        criterion:cuda()
    end

    local parametersD, gradParametersD = netD:getParameters()
    local parametersG, gradParametersG = netG:getParameters()

    local generateNoise = function()
        if opt.noise == 'uniform' then -- regenerate random noise
            noise:uniform(-1, 1)
        elseif opt.noise == 'normal' then
            noise:normal(0, 1)
        end
    end

    -- create closure to evaluate f(X) and df/dX of discriminator
    local fDx = function(x)
        netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

        gradParametersD:zero()

        -- train with real
        label:fill(real_label)

        local batch = data.random_batch(opt.batchSize)
        input:copy(batch.data)
        classes:copy(batch.labels)
        local combined_input = {input, classes}

        local output = netD:forward(combined_input)
        local errD_real = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        netD:backward(combined_input, df_do)

        -- train with fake
        label:fill(fake_label)

        generateNoise()
        local fake = netG:forward(noise)
        input:copy(fake)

        for i = 1, opt.batchSize do
            for j = 1, opt.labelCount do
                classes[i][j] = noise[i][j][1][1]
            end
        end

        local combined_input = {input, classes}

        local output = netD:forward(combined_input)
        local errD_fake = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        netD:backward(combined_input, df_do)

        errD = errD_real + errD_fake

        return errD, gradParametersD
    end

    -- create closure to evaluate f(X) and df/dX of generator
    local fGx = function(x)
        netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

        gradParametersG:zero()

        generateNoise()

        local fake = netG:forward(noise)
        input:copy(fake)
        label:fill(real_label)

        for i = 1, opt.batchSize do
            for j = 1, opt.labelCount do
                classes[i][j] = noise[i][j][1][1]
            end
        end

        local combined_input = {input, classes}

        local output = netD:forward(combined_input)

        errG = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        local df_dg = netD:updateGradInput(combined_input, df_do)

        netG:backward(noise, df_dg[1])
        return errG, gradParametersG
    end

    -- Training functions
    local function trainD()
        parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
        parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them

        optim.adam(fDx, parametersD, optimStateD)
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
            trainD()
        end

        trainG()

        if not callback(epoch, epoch_tm:time().real, errD, errG) then
            break
        end
    end
end
