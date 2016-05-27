require 'torch'
require 'nn'
require 'audio'
require "../lib/gan.lua"
require "../lib/dataset.lua"
require "../lib/audio_utils.lua"

init(10)

opt = get_default_opt()

opt.nz = {200, 1, 1}               -- #  of dim for Z
opt.n = {1, 1024, 1}               -- #  of gen filters in first conv layer
opt.maxEpochs = 200
opt.maxSubEpochs = 1000
opt.lr = 0.0001            -- initial learning rate for adam
opt.beta1 = 0.5            -- momentum term of adam
opt.ntrain = math.huge     -- #  of examples per epoch. math.huge for full dataset
opt.gpu = 0                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
opt.name = 'experiment1'
opt.noise = 'uniform'      -- uniform / normal
opt.data = 100
opt.batchSize = 10
opt.trainingAlgorithm = 'original'
opt.errThrD = 0.500
opt.subEpochPrintInterval = 10

local function saw_tooth(dim, freq, frame_rate)
    local values = torch.Tensor(table.unpack(dim))
    local a = (frame_rate/freq)
    for x = 1, dim[2] do
        values[1][x][1] = 2 * (x/a - math.floor(1/2 + x/a))
    end
    return values;
end

local function triangle(dim, freq, frame_rate)
    return saw_tooth(dim, freq, frame_rate):apply(function(x) return math.abs(x) * 2 - 1 end);
end

local function sine(dim, freq, frame_rate)
    local values = torch.Tensor(table.unpack(dim))
    for x = 1, dim[2] do
        values[1][x][1] = math.sin(2 * math.pi * (freq*x/frame_rate))
    end
    return values;
end

local function square(dim, freq, frame_rate)
    return sine(dim, freq, frame_rate):apply(function(x) if x > 0 then return 0.99 else return -0.99 end end)
end

-- create data loader
local data = torch.Tensor(opt.data, table.unpack(opt.n))
for i = 1, opt.data do
		data[i] = sine(opt.n, 500+i*100, 44100);
		--data[i] = saw_tooth(opt.n, 500+i*100, 44100);
		--data[i] = triangle(opt.n, 500+i*100, 44100);
		--data[i] = square(opt.n, 500+i*100, 44100);
end

local dataset = tensor_to_dataset(data)

local netG = nn.Sequential()
local Gkern = 8;
local Gstep = 4;

local ndf = 64
local ngf = 64
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(opt.nz[1], ngf*8, 1, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*16) x 4 x 1
--netG:add(nn.SpatialFullConvolution(ngf*16, ngf*8, 1, Gkern, 1, Gstep, 0, Gstep/2))
--netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 1
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 1, Gkern, 1, Gstep, 0, Gstep/2))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 1
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 1, Gkern, 1, Gstep, 0, Gstep/2))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 1
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf * 1, 1, Gkern, 1, Gstep, 0, Gstep/2))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 1
--netG:add(nn.SpatialFullConvolution(ngf, 1, 1, 5, 1, 1, 0, Gstep/2))
netG:add(nn.SpatialFullConvolution(ngf, 1, 1, Gkern, 1, Gstep, 0, Gstep/2))
netG:add(nn.Tanh())
-- state size: 1 x 64 x 1

local netD = nn.Sequential()
local Dkern = 8;
local Dstep = 4;

-- input is 1 x 64 x 1
netD:add(nn.SpatialConvolution(1, ndf, 1, Dkern, 1, Dstep, 0, Dstep/2))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 1
netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, Dkern, 1, Dstep, 0, Dstep/2))
netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
----- state size: (ndf*2) x 16 x 1
netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 1, Dkern, 1, Dstep, 0, Dstep/2))
netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
------ state size: (ndf*4) x 8 x 1
netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 1, Dkern, 1, Dstep, 0, Dstep/2))
netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
------ state size: (ndf*4) x 8 x 1
netD:add(nn.SpatialConvolution(ndf * 8, ndf * 16, 1, Dkern, 1, Dstep, 0, Dstep/2))
netD:add(nn.SpatialBatchNormalization(ndf * 16)):add(nn.LeakyReLU(0.2, true))
------ state size: (ndf*4) x 8 x 1
netD:add(nn.SpatialConvolution(ndf * 16, ndf * 32, 1, Dkern, 1, Dstep, 0, Dstep/2))
netD:add(nn.SpatialBatchNormalization(ndf * 32)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*8) x 4 x 1
netD:add(nn.SpatialConvolution(ndf * 32, 1, 1, 1))
netD:add(nn.Sigmoid())
-------- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

local function callback(epoch)
    local n = torch.Tensor(10, table.unpack(opt.nz));
    if opt.noise == 'uniform' then -- regenerate random noise
        n:uniform(-1, 1)
    elseif opt.noise == 'normal' then
        n:normal(0, 1)
    end
    if opt.gpu > 0 then
        n = n:cuda();
    end
    paths.mkdir('results' .. epoch)
    local r = netG:forward(n);
    for i = 1, 10 do
        local signal = torch.Tensor(opt.n[2])
        for j = 1, opt.n[2] do
            -- MAKES TOTAL SENS
            signal[j] = r[i][1][j][1]
        end

        print("Saving "..i)
        local m = filter_audio(filter_audio(filter_audio(normalize_audio(signal), 5, average_filter_fun), 5, average_filter_fun), 5, average_filter_fun)
        save_audio(m, "results"..epoch.."/sine"..i..".wav")
        save_audio(signal, "results"..epoch.."/original"..i..".wav")
    end
end

--opt.trainingAlgorithm = 'original'
--train(netG, netD, dataset, opt, false)
--callback(opt.maxEpochs)

train(netG, netD, dataset, opt, false)
callback(opt.maxEpochs)
