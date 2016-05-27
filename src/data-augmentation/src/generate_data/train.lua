require 'torch'
require 'nn'
require "../lib/gan.lua"
require "../lib/dataset.lua"

if #arg ~= 6 then
	print("Usage: th train.lua <gpu> <name> <datasetDir> <datasetMode ('lazy'/'memory')> <epochs> <checkpointInterval>")
	return
end

local gpu = tonumber(arg[1])
local name = arg[2]
local datasetDir = arg[3]
local datasetMode = arg[4]
local epochs = tonumber(arg[5])
local checkpointInterval = tonumber(arg[6])


init(10)

opt = get_default_opt()

opt.batchSize = 64
opt.nz = {50, 1, 1}
opt.maxEpochs = epochs
opt.lr = 0.0002
opt.beta1 = 0.5
opt.gpu = gpu
opt.name = name
opt.noise = 'normal'
opt.n = {1, 32, 32}
opt.checkpointInterval = checkpointInterval
opt.trainingAlgorithm = 'original'

print("Loading data ...")
if datasetMode == 'lazy' then
    dataset = load_image_dir(datasetDir)
else
    dataset = load_image_dir_in_memory(datasetDir)
end

print("Constructing networks ...")
local nc = 1
local nz = opt.nz[1]
local ndf = 64
local ngf = 64

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, nc, 4, 4, 2, 2, 1, 1))
-- state size: (nc) x 32 x 32
netG:add(nn.Tanh())

weights_init(netG)


local netD = nn.Sequential()

-- state size: (nc) x 32 x 32
netD:add(SpatialConvolution(nc, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

weights_init(netD)


print("Training ...")
train(netG, netD, dataset, opt, false)
