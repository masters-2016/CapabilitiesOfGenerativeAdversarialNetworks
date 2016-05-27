require 'torch'
require 'nn'
require 'audio'
require "../lib/gan.lua"
require "../lib/dataset.lua"
require "../lib/audio_utils.lua"

if #arg ~= 3 then
	print("Usage: th gan_short.lua <wave_type> <network> <name>")
    print("    Wave types: sine, saw_tooth, triangle, sqaure")
    print("    Network types: mlp, cnn, fcnn")
	return
end

local wave_type = arg[1]
local network = arg[2]
local name = arg[3]

init(10)

opt = get_default_opt()

opt.nz = {30, 1, 1}               -- #  of dim for Z
opt.n = {1, 64, 1}               -- #  of gen filters in first conv layer
opt.maxEpochs = 25000
opt.lr = 0.0005            -- initial learning rate for adam
opt.beta1 = 0.5            -- momentum term of adam
opt.gpu = 0                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
opt.name = name
opt.noise = 'uniform'      -- uniform / normal
opt.data = 10000
opt.batchSize = 64
opt.trainingAlgorithm = 'original'
opt.checkpointInterval = 1000

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
    local freq = 500+i*(500/opt.data)
    if wave_type == "sine" then
		data[i] = sine(opt.n, freq, 44100);
    elseif wave_type == "saw_tooth" then
		data[i] = saw_tooth(opt.n, freq, 44100);
    elseif wave_type == "triangle" then
		data[i] = triangle(opt.n, freq, 44100);
    elseif wave_type == "square" then
		data[i] = square(opt.n, freq, 44100);
    end
end

paths.mkdir(("%s_dataset"):format(wave_type))
for i = 1, opt.data do
    local signal = torch.Tensor(opt.n[2])
    for j = 1, opt.n[2] do
        -- MAKES TOTAL SENS
        signal[j] = data[i][1][j][1]
    end
    save_audio(signal, ("%s_dataset/%06d.wav"):format(wave_type, i))
end


local dataset = tensor_to_dataset(data)

local netG = nn.Sequential()
local kern = 4;
local step = 2;

local ndf = 64
local ngf = 64
if network == "fcnn" then 
    -- input is Z, going into a convolution
    netG:add(nn.SpatialFullConvolution(opt.nz[1], ngf*8, 1, 4))
    netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    -- state size: (ngf*8) x 4 x 1
    netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 1, kern, 1, step, 0, step/2))
    netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 1
    netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 1, kern, 1, step, 0, step/2))
    netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*2) x 16 x 1
    netG:add(nn.SpatialFullConvolution(ngf * 2, ngf * 1, 1, kern, 1, step, 0, step/2))
    netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf) x 32 x 1
    netG:add(nn.SpatialFullConvolution(ngf, 1, 1, kern, 1, step, 0, step/2))
    netG:add(nn.Tanh())
    -- state size: 1 x 64 x 1
elseif network == "cnn" then
    -- input is Z, going into a convolution
    netG:add(nn.SpatialFullConvolution(opt.nz[1], ngf*8, 1, 4))
    netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    -- state size: (ngf*8) x 4 x 1
    netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 1, kern, 1, step, 0, step/2))
    netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 1
    netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 1, kern, 1, step, 0, step/2))
    netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*2) x 16 x 1
    netG:add(nn.SpatialFullConvolution(ngf * 2, ngf * 1, 1, kern, 1, step, 0, step/2))
    netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf) x 32 x 1
    netG:add(nn.SpatialFullConvolution(ngf, 1, 1, kern, 1, step, 0, step/2))
    netG:add(nn.View(64))
    netG:add(nn.Linear(64, 64))
    netG:add(nn.Tanh())
    netG:add(nn.Linear(64, 64))
    netG:add(nn.View(1, 64, 1))
    netG:add(nn.Tanh())
    -- state size: 1 x 64 x 1
elseif network == "mlp" then
    netG:add(nn.View(opt.nz[1]))
    netG:add(nn.Linear(opt.nz[1], 64))
    netG:add(nn.Tanh())
    netG:add(nn.Linear(64, 64))
    netG:add(nn.Tanh())
    netG:add(nn.Linear(64, 64))
    netG:add(nn.Tanh())
    netG:add(nn.Linear(64, 64))
    netG:add(nn.Tanh())
    netG:add(nn.Linear(64, 64))
    netG:add(nn.View(1, 64, 1))
    netG:add(nn.Tanh())
end

local netD = nn.Sequential()

-- input is 1 x 64 x 1
netD:add(nn.SpatialConvolution(1, ndf, 1, kern, 1, step, 0, step/2))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 1
netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, kern, 1, step, 0, step/2))
netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
----- state size: (ndf*2) x 16 x 1
netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 1, kern, 1, step, 0, step/2))
netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
------ state size: (ndf*4) x 8 x 1
netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 1, kern, 1, step, 0, step/2))
netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*8) x 4 x 1
netD:add(nn.SpatialConvolution(ndf * 8, 1, 1, 4))
netD:add(nn.Sigmoid())
---- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

train(netG, netD, dataset, opt, false)
