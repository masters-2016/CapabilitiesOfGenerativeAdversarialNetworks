require 'image'
require 'nn'
require '../lib/supervised_dataset'
require "../lib/audio_utils.lua"
util = paths.dofile('../lib/util.lua')

function get_default_generation_opt()
    local opt = {
        noisetype = 'uniform', -- type of noise distribution (uniform / normal).
        net = '',              -- path to the generator network
        dirname = '',          -- The path to the directory with the generation definition file
        filename = '',         -- The name of the generation definition file
        nz = 100,              
        labelCount = 1
    }
    for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

    return opt
end

function generate_image(netFile, seed, count, nz, labels, filename)
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(seed)

    local noise = torch.Tensor(count, nz + #labels, 1, 1):uniform(-1, 1)


    for i = 1, count do
        for j = 1, #labels do
            noise[i][j][1][1] = labels[j]
        end
    end

    local net = util.load(netFile, 0)
    net:float()
    util.optimizeInferenceMemory(net)

    local images = net:forward(noise)
    images:add(1):mul(0.5)
    image.save(filename, image.toDisplayTensor(images))
end

function generate_audio(netFile, seed, nz, output_size, filename)
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(seed)

    local noise = torch.Tensor(1, nz, 1, 1):uniform(-1, 1)

    local net = util.load(netFile, 0)
    net:float()
    util.optimizeInferenceMemory(net)

    local audio = net:forward(noise)
    local signal = torch.Tensor(output_size)
    for j = 1, output_size do
        -- DOES REALLY MAKE TOTAL SENS (WE HAVE THE BEST WORDS)
        signal[j] = audio[1][1][j][1]
    end
    save_audio(signal, filename)
end

function generate_images(opt)
    print(opt)

    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed( math.random(10000) )

    local noise = torch.Tensor(1, opt.nz + opt.labelCount, 1, 1)

    local net = util.load(opt.net, 0)
    net:float()

    -- for older models, there was nn.View on the top
    -- which is unnecessary, and hinders convolutional generations.
    if torch.type(net:get(1)) == 'nn.View' then
        net:remove(1)
    end

    print(net)

    -- a function to setup double-buffering across the network.
    -- this drastically reduces the memory needed to generate samples
    util.optimizeInferenceMemory(net)

    local generations = get_filenames_and_labels(opt.dirname .. '/' .. opt.filename)

    for i = 1, #generations do
        local gen = generations[i]

        if opt.noisetype == 'uniform' then
            noise:uniform(-1, 1)
        elseif opt.noisetype == 'normal' then
            noise:normal(0, 1)
        end

        for j = 1, opt.labelCount do
            noise[1][j][1][1] = gen.labels[j]
        end

        local images = net:forward(noise)
        images:add(1):mul(0.5)
        image.save(opt.dirname .. '/' .. gen.filename, images[1])
    end
end
