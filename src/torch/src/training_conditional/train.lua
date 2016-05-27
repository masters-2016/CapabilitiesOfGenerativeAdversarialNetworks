require 'torch'
require 'nn'
require 'nngraph'
require "../lib/gan.lua"
require "../lib/mlp.lua"
require "../lib/supervised_dataset.lua"

init(10)

if #arg ~= 5 then
	print("Usage: th train.lua <name> <gpu> <labelCount> <datasetDir> <datasetMode ('lazy'/'memory')>")
	return
end

local name = arg[1]
local gpu = tonumber(arg[2])

local labelCount = tonumber(arg[3])
local noiseCount = 100 - labelCount

local datasetDir = arg[4]
local datasetMode = arg[5]

opt = get_default_opt()

opt.batchSize = 64
opt.nz = {noiseCount, 1, 1}
opt.maxEpochs = 25000
opt.lr = 0.0002
opt.beta1 = 0.5
opt.gpu = gpu
opt.name = name
opt.noise = 'uniform'
opt.n = {3, 64, 64}
opt.checkpointInterval = 2500
opt.trainingAlgorithm = 'conditional'
opt.labelCount = labelCount

print("Loading data ...")

if datasetMode == 'lazy' then
	dataset = load_supervised_image_dataset(datasetDir, 'labels.txt')
else
	dataset = load_supervised_image_dataset_in_memory(datasetDir, 'labels.txt')
end

print("Constructing networks ...")
local nc = 3
local nz = opt.nz[1] + opt.labelCount
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


local netD_image = nn.Sequential()

-- input is (nc) x 64 x 64
netD_image:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD_image:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD_image:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD_image:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD_image:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD_image:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD_image:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD_image:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD_image:add(SpatialConvolution(ndf * 8, 64, 4, 4))
netD_image:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD_image:add(nn.View(64):setNumInputDims(3))
-- state size: 1

netD_image:apply(weights_init)


local netD_image_node = netD_image()
local netD_classes_node = nn.Identity()()
local netD_combiner = nn.JoinTable(2)({netD_image_node, netD_classes_node})
local netD_output = nn.Sigmoid()(nn.Linear(64+labelCount, 1)(netD_combiner))

netD = nn.gModule({netD_image_node, netD_classes_node}, {netD_output})

--graph.dot(netD.fg, 'MLP', 'TestMLP')

print("Training GAN ...")
train(netG, netD, dataset, opt, false)
