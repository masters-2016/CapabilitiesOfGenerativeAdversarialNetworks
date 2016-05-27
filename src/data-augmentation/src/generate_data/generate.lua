require 'torch'
require 'image'
require 'nn'
util = paths.dofile('../lib/util.lua')

torch.setdefaulttensortype('torch.FloatTensor')

local checkpoint_dir = 'checkpoints'
local gen_dir = 'gen'

paths.mkdir(gen_dir)

labels = {}
labels[0] = "1,0,0,0,0,0,0,0,0,0"
labels[1] = "0,1,0,0,0,0,0,0,0,0"
labels[2] = "0,0,1,0,0,0,0,0,0,0"
labels[3] = "0,0,0,1,0,0,0,0,0,0"
labels[4] = "0,0,0,0,1,0,0,0,0,0"
labels[5] = "0,0,0,0,0,1,0,0,0,0"
labels[6] = "0,0,0,0,0,0,1,0,0,0"
labels[7] = "0,0,0,0,0,0,0,1,0,0"
labels[8] = "0,0,0,0,0,0,0,0,1,0"
labels[9] = "0,0,0,0,0,0,0,0,0,1"

labelfile =  io.open("gen/labels.txt","w")

local x = 0
for i = 0, 9 do

    local net = util.load(("%s/%d_5000_net_G.t7"):format(checkpoint_dir, i), 0)
    net:float()
    util.optimizeInferenceMemory(net)
    
    for j = 1, 165 do
        local noise = torch.Tensor(100, 50, 1, 1):uniform(-1, 1)
        local images = net:forward(noise)
        --images:add(1):mul(0.5)

        for y = 1, 100 do
            local filename = ("%s/%06d.png"):format(gen_dir, x)
            image.save(filename, images[y])

            labelfile:write(("%06d.png %s\n"):format(x, labels[i]))

            x = x + 1
        end
    end
end
