require 'torch'
require 'nn'
require 'optim'

util = paths.dofile('../lib/util.lua')

function train_original_supervised2(netG, netD, data, opt, callback, netC)

    local function weights_init(m)
        local name = torch.type(m)
        if name:find('Convolution') then
            m.weight:normal(0.0, 0.02)
            m.bias:fill(0)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(1.0, 0.02) end
            if m.bias then m.bias:fill(0) end
        elseif name:find('Linear') then
            if m.weight then m.weight:normal(1.0, 0.02) end
            if m.bias then m.bias:fill(0) end
        end
    end

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
    optimStateC = {
        learningRate = opt.lr,
        beta1 = opt.beta1,
    }

    local input = torch.Tensor(opt.batchSize, unpack(opt.n))

    local noise_count = opt.nz
    noise_count[1] = noise_count[1] + opt.labelCount
    local noise = torch.Tensor(opt.batchSize, unpack(noise_count))

    local label = torch.Tensor(opt.batchSize)
    local classes = torch.Tensor(opt.batchSize, opt.labelCount)

    local errD, errG, errC
    local epoch_tm = torch.Timer()
    local tm = torch.Timer()
    local data_tm = torch.Timer()

    if opt.gpu > 0 then
        require 'cunn'
        cutorch.setDevice(opt.gpu)

        input = input:cuda()
        noise = noise:cuda()
        label = label:cuda()
        classes = classes:cuda()

        netG = util.cudnn(netG)
        netD = util.cudnn(netD)
        netC = util.cudnn(netC)

        netD:cuda()
        netG:cuda()
        netC:cuda()
        criterion:cuda()
    end

    local parametersD, gradParametersD = netD:getParameters()
    local parametersG, gradParametersG = netG:getParameters()
    local parametersC, gradParametersC = netC:getParameters()

    noise_vis = noise:clone()
    if opt.noise == 'uniform' then
        noise_vis:uniform(-1, 1)
    elseif opt.noise == 'normal' then
        noise_vis:normal(0, 1)
    end

    -- create closure to evaluate f(X) and df/dX of discriminator
    local fDx = function(x)
        netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

        gradParametersD:zero()

        -- train with real
        data_tm:stop()
        local batch = data.random_batch(opt.batchSize).data
        input:copy(batch)
        label:fill(real_label)

        local output = netD:forward(input)
        --print(output:size())
        local errD_real = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        netD:backward(input, df_do)

        -- train with fake
        if opt.noise == 'uniform' then -- regenerate random noise
            noise:uniform(-1, 1)
        elseif opt.noise == 'normal' then
            noise:normal(0, 1)
        end

        local fake = netG:forward(noise)
        --print(fake:size())
        input:copy(fake)
        label:fill(fake_label)

        local output = netD:forward(input)
        local errD_fake = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        netD:backward(input, df_do)

        errD = errD_real + errD_fake

        return errD, gradParametersD
    end

    -- create closure to evaluate f(X) and df/dX of classifier
    local fCx = function(x)
        netC:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

        gradParametersC:zero()

        -- train with real
        data_tm:stop()
        local batch = data.random_batch(opt.batchSize)
        input:copy(batch.data)
        classes:copy(batch.labels)

        local output = netC:forward(input)
        errC = criterion:forward(output, classes)
        local df_do = criterion:backward(output, classes)
        netC:backward(input, df_do)

        return errC, gradParametersC
    end

    -- create closure to evaluate f(X) and df/dX of generator
    local fGx = function(x)
        netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netC:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        
        gradParametersG:zero()

        -- Train against D
        if opt.noise == 'uniform' then -- regenerate random noise
            noise:uniform(-1, 1)
        elseif opt.noise == 'normal' then
            noise:normal(0, 1)
        end

        local fake = netG:forward(noise)
        input:copy(fake)

        label:fill(real_label) -- fake labels are real for generator cost

        netD:forward(input)
        local output = netD.output

        local errG_D = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        local df_dg = netD:updateGradInput(input, df_do)

        netG:backward(noise, df_dg)

        -- Train against C
        if opt.noise == 'uniform' then -- regenerate random noise
            noise:uniform(-1, 1)
        elseif opt.noise == 'normal' then
            noise:normal(0, 1)
        end

        local fake = netG:forward(noise)
        input:copy(fake)

        for i = 1, opt.batchSize do
            for j = 1, opt.labelCount do
                classes[i][j] = noise[i][j]
            end
        end

        netC:forward(input)
        local output = netC.output

        local errG_C = criterion:forward(output, classes)
        local df_do = criterion:backward(output, classes)
        local df_dg = netC:updateGradInput(input, df_do)

        netG:backward(noise, df_dg)

        -- Combine and return the errors
        errG = errG_D + errG_C
        return errG, gradParametersG
    end

    -- Training functions
    local function trainD()
        parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
        parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them

        optim.adam(fDx, parametersD, optimStateD)
    end

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

        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        for x = 1, opt.k do
            trainD()
            trainC()
        end

        -- (2) Update G network: maximize log(D(G(z)))
        trainG()

        if not callback(epoch, epoch_tm:time().real, errD, errG, errC) then
            break
        end
    end
end
