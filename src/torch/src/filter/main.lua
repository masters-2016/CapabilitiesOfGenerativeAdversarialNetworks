require 'torch'
require 'nn'

require 'utils.lua'
require 'dataset.lua'
require 'training.lua'

print('Constructing network ...')
local net = nn.Sequential()

net:add(nn.SpatialConvolution(3, 256, 3, 3, 1, 1, 1, 1))
net:add(nn.LeakyReLU(0.2, true))

net:add(nn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
net:add(nn.LeakyReLU(0.2, true))

net:add(nn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
net:add(nn.LeakyReLU(0.2, true))

net:add(nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1))
net:add(nn.LeakyReLU(0.2, true))

net:add(nn.SpatialConvolution(32, 16, 3, 3, 1, 1, 1, 1))
net:add(nn.LeakyReLU(0.2, true))

net:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
net:add(nn.LeakyReLU(0.2, true))

net:add(nn.SpatialConvolution(8, 3, 3, 3, 1, 1, 1, 1))
net:add(nn.LeakyReLU(0.2, true))

net:add(nn.Sigmoid())


print('Defining options ...')
options = default_options()
options.learningRate = 0.0002
options.batchSize = 64
options.epochs = 25000
options.gpu = 1


print('Loading training data ...')
x, y = load_dataset('training_dataset', 10000, 'png')


print('Loading test data ...')
x_test, y_test = load_dataset('real_images', 4, 'jpg')

if options.gpu then
    require 'cunn'
    cutorch.setDevice(options.gpu)
    x_test = x_test:cuda()
    y_test = y_test:cuda()
end


print('Defining callback ...')
paths.mkdir('out')

local function callback(epoch)
    if epoch % 100 == 0 then
        outdir = ('out/%06d'):format(epoch)
        paths.mkdir(outdir)

        output = net:forward(x_test)

        images = torch.Tensor(3, 3, 64, 64)
        for i = 1, x_test:size(1) do
            images[1]:copy(x_test[i])
            images[2]:copy(output[i])
            images[3]:copy(y_test[i])

            filename = ('%s/%06d.png'):format(outdir, i)
            save_images(filename, images)
        end
    end
end


print('Training network ...')
train(net, x, y, options, callback)
