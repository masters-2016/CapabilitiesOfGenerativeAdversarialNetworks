require 'torch'
require 'nn'
require "../lib/gan.lua"
require "../lib/mlp.lua"
require "../lib/dataset.lua"
util = paths.dofile('../lib/util.lua')

init(10)

if #arg ~= 9 then
	print("Usage: th train.lua <name> <gpu> <batchSize> <datasetDir> <datasetSize> <datasetMode ('lazy'/'memory')> <maxEpochs> <checkpointInterval> <maxTimeInSeconds>")
	return
end

local name = arg[1]
local gpu = tonumber(arg[2])

local batchSize = tonumber(arg[3])

local datasetDir = arg[4]
local datasetSize = tonumber(arg[5])
local datasetMode = arg[6]

local maxEpochs = tonumber(arg[7])
local checkpointInterval = tonumber(arg[8])
local maxTimeInSeconds = tonumber(arg[9])

opt = get_default_opt()

opt.batchSize = batchSize
opt.nz = {100, 1, 1}
opt.maxEpochs = maxEpochs
opt.maxTimeInSeconds = maxTimeInSeconds
opt.lr = 0.0002
opt.beta1 = 0.5
opt.gpu = gpu
opt.name = name
opt.noise = 'uniform'
opt.n = {3, 64, 64}
opt.checkpointInterval = checkpointInterval
opt.trainingAlgorithm = 'original'

print("Loading data ...")

if datasetMode == 'lazy' then
	dataset = load_image_dir(datasetDir, datasetSize)
else
	dataset = load_image_dir_in_memory(datasetDir, datasetSize)
end

print("Constructing networks ...")
local nc = 3
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
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)


local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
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

netD:apply(weights_init)

print("Training GAN ...")
train(netG, netD, dataset, opt, false)

print("Saving trained network ...")
paths.mkdir('checkpoints')
util.save('checkpoints/' .. opt.name .. '_net_G.t7', netG, opt.gpu)
