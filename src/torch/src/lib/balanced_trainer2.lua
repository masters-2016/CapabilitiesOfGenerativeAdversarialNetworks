require 'torch'
require 'nn'
require 'optim'

util = paths.dofile('../lib/util.lua')

function train_balanced2(netG, netD, data, opt, callback)

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
    local noise = torch.Tensor(opt.batchSize, unpack(opt.nz))
    local label = torch.Tensor(opt.batchSize)
    local errD, errG
    local epoch_tm = torch.Timer()
    local tm = torch.Timer()
    local data_tm = torch.Timer()

    if opt.gpu > 0 then
        require 'cunn'
        cutorch.setDevice(opt.gpu)
        input = input:cuda();  noise = noise:cuda();  label = label:cuda()
        netG = util.cudnn(netG);     netD = util.cudnn(netD)
        netD:cuda();           netG:cuda();           criterion:cuda()
    end

    local parametersD, gradParametersD = netD:getParameters()
    local parametersG, gradParametersG = netG:getParameters()
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fDx = function(x)
        netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

        gradParametersD:zero()

        -- train with real
        data_tm:stop()
        local batch = data.random_batch(opt.batchSize)
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

    -- create closure to evaluate f(X) and df/dX of generator
    local fGx = function(x)
        netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
        netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

        gradParametersG:zero()

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

        errG = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        local df_dg = netD:updateGradInput(input, df_do)

        netG:backward(noise, df_dg)
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

        local term = false
        local c = 0

        tm:reset()

        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        trainD()
        if opt.subEpochPrintInterval then
            print("Subepoch D, 0 -> errD = " .. errD)
        end

        while opt.errThrD and errD > opt.errThrD do
            trainD()

            c = c + 1
            if opt.subEpochPrintInterval and c % opt.subEpochPrintInterval == 0 then
                print("Subepoch D, " .. c .. " -> errD = " .. errD)
            end
                
            if opt.maxSubEpochs and c > opt.maxSubEpochs then
                term = true
                break
            end
        end

        if opt.subEpochPrintInterval then
            print("Subepoch D, " .. c .. " -> errD = " .. errD)
        end

        -- (2) Update G network: maximize log(D(G(z)))
        trainG()
        if opt.subEpochPrintInterval then
            print("Subepoch G, 0 -> errG = " .. errG)
        end

        c = 0
        while opt.errThrG and errG > opt.errThrG do
            trainG()

            c = c + 1
            if opt.subEpochPrintInterval and c % opt.subEpochPrintInterval == 0 then
                print("Subepoch G, " .. c .. " -> errG = " .. errG)
            end

            if opt.maxSubEpochs and c > opt.maxSubEpochs then
                term = true
                break
            end
        end

        if opt.subEpochPrintInterval then
            print("Subepoch G, " .. c .. " -> errG = " .. errG)
        end

        if not callback(epoch, epoch_tm:time().real, errD, errG) then
            break
        end
    end
end
