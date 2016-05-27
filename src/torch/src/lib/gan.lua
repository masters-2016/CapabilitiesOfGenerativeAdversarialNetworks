require 'torch'
require 'nn'

require '../lib/balanced_trainer'
require '../lib/balanced_trainer2'
require '../lib/original_trainer'
require '../lib/original_trainer_supervised'
require '../lib/original_trainer_supervised2'
require '../lib/original_trainer_supervised_binary_labels'
require '../lib/original_trainer_supervised_binary_labels_separate_c'
require '../lib/original_trainer_supervised_no_discriminator'
require '../lib/original_trainer_conditional'

util = paths.dofile('../lib/util.lua')

function get_default_opt()
    local opt = {
        nz = {9},                       -- #  of dim for Z
        lr = 0.001,                     -- initial learning rate for adam
        beta1 = 0.5,                    -- momentum term of adam
        ntrain = math.huge,             -- #  of examples per epoch. math.huge for full dataset
        gpu = 0,                        -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
        name = 'experiment1',
        noise = 'normal',               -- uniform / normal
        batchSize = 10,
        errThrD = 0.001,                -- float or false
        errThrG = false,                -- float or false
        checkpointInterval = false,     -- int or false
        checkpointD = false,            -- true or false
        checkpointC = false,            -- true or false
        subEpochPrintInterval = false,  -- int or false
        maxEpochs = false,              -- int or false
        maxSubEpochs = false,           -- int or false
        k = 1,
        dTail = 0,
        dTailAlgorithm = 'recent',      -- 'recent' / 'sample'
        trainingAlgorithm = 'original', -- 'balanced' / 'balanced2' / 'original' / 'supervised' / 'supervised2' / 'supervised_binary_labels' / 'train_original_supervised_binary_labels_separate_c' / 'supervised_no_discriminator' / 'conditional'
        labelCount = 0,                 -- int. Number of labels in the supervised dataset
        maxTimeInSeconds = false
    }

    -- one-line argument parser. parses enviroment variables to override the defaults
    for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

    return opt
end

function init(manualSeed)
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.setnumthreads(1)
    torch.manualSeed(manualSeed)
end

function train(netG, netD, data, opt, callback, netC)
    print(opt)

    local total_tm = torch.Timer()
    total_tm:reset()

    local function wrapped_callback(epoch, epoch_time, errD, errG, errC)
        paths.mkdir('checkpoints')
        if opt.checkpointInterval and (epoch % opt.checkpointInterval) == 0 then
            util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG, opt.gpu)

            if opt.checkpointD and netD then
                util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD, opt.gpu)
            end

            if opt.checkpointC and netC then
                util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_C.t7', netC, opt.gpu)
            end
        end

        if errC then
            print(('epoch %d,epoch_time %.3f,total_time %.3f,errD %.6f,errG %.6f,errC %.6f'):format(
            epoch, epoch_time, total_tm:time().real, errD, errG, errC))
        else
            print(('epoch %d,epoch_time %.3f,total_time %.3f,errD %.6f,errG %.6f'):format(
            epoch, epoch_time, total_tm:time().real, errD, errG))
        end

        if callback then
            callback(epoch)
        end

        if opt.maxTimeInSeconds and total_tm:time().real > opt.maxTimeInSeconds then
            util.save('checkpoints/' .. opt.name .. '_final_net_G.t7', netG, opt.gpu)
            return false;
        end

        return true;
    end

    if opt.trainingAlgorithm == 'balanced' then
        train_balanced(netG, netD, data, opt, wrapped_callback)
    elseif opt.trainingAlgorithm == 'balanced2' then
        train_balanced2(netG, netD, data, opt, wrapped_callback)
    elseif opt.trainingAlgorithm == 'original' then
        train_original(netG, netD, data, opt, wrapped_callback)
    elseif opt.trainingAlgorithm == 'subset' then
        train_original_subset(netG, netD, data, opt, wrapped_callback)
    elseif opt.trainingAlgorithm == 'supervised' then
        train_original_supervised(netG, netD, data, opt, wrapped_callback)
    elseif opt.trainingAlgorithm == 'supervised2' then
        train_original_supervised2(netG, netD, data, opt, wrapped_callback, netC)
    elseif opt.trainingAlgorithm == 'supervised_binary_labels' then
        train_original_supervised_binary_labels(netG, netD, data, opt, wrapped_callback)
    elseif opt.trainingAlgorithm == 'train_original_supervised_binary_labels_separate_c' then
        train_original_supervised_binary_labels_separate_c(netG, netD, data, opt, wrapped_callback, netC)
    elseif opt.trainingAlgorithm == 'supervised_no_discriminator' then
        train_original_supervised_no_discriminator(netG, netD, data, opt, wrapped_callback, netC)
    elseif opt.trainingAlgorithm == 'conditional' then
        train_original_conditional(netG, netD, data, opt, wrapped_callback)
    end
end

function weights_init(m)
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
