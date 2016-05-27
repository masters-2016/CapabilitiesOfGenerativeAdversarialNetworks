require 'torch'
require 'nn'

util = paths.dofile('../lib/util.lua')

function train_mlp(net, dataset, opt, callback)
    --local criterion = nn.BCECriterion()
    local criterion = nn.MSECriterion()

    local input = torch.Tensor(opt.batchSize, unpack(opt.n))
    local labels = torch.Tensor(opt.batchSize, table.unpack(opt.labelCount))

    if opt.gpu then
        require 'cunn'
        cutorch.setDevice(opt.gpu)

        input = input:cuda()
        labels = labels:cuda()

        net = util.cudnn(net)
        net:cuda()

        criterion:cuda()
    end

    local batchCount = math.ceil(dataset:count() / opt.batchSize)

    for e = 1,opt.maxEpochs do
        
        acc_err = 0
        for i = 1, batchCount do
            local batch = dataset.random_batch(opt.batchSize)

            input:copy(batch.data)
            labels:copy(batch.labels)

            -- feed it to the neural network and the criterion
            acc_err = acc_err + criterion:forward(net:forward(input), labels)

            -- train over this example in 3 steps
            -- (1) zero the accumulation of the gradients
            net:zeroGradParameters()

            -- (2) accumulate gradients
            net:backward(input, criterion:backward(net.output, labels))

            -- (3) update parameters with a 0.01 learning rate
            net:updateParameters(opt.lr)
        end

        local err = acc_err / batchCount
        print(('%d: %.6f'):format(e, err))

        if callback then
            callback(e, err)
        end
    end

end

function get_default_mlp_opt()
    local opt = {
        lr = 0.001,     	-- learning rate
        maxEpochs = 25,
		batchSize = 20,
		n = {3, 64, 64}, 	-- dimensions of single input
		labelCount = {1},
        gpu = false
    }

    -- one-line argument parser. parses enviroment variables to override the defaults
    for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

    return opt
end
