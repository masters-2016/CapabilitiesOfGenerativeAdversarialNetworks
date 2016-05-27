require "gan.lua"
require "dataset.lua"

init(5)

local opt = get_default_opt()
--opt.subEpochPrintInterval = 100
opt.errThrD = 0.001
opt.n = {4}
opt.nz = {10}
--opt.dTail = 0
--opt.dTailAlgorithm = 'recent'
--opt.dTailAlgorithm = 'sample'
opt.lr = 0.001
--opt.trainingAlgorithm = 'balanced'
opt.trainingAlgorithm = 'original'
opt.maxEpochs = 500
opt.maxSubEpochs = 10000

local netG = nn.Sequential()
local HUs = opt.nz[1];
netG:add(nn.Linear(opt.nz[1], HUs))
netG:add(nn.Add(HUs,false))
netG:add(nn.Linear(HUs, opt.n[1]))

local netD = nn.Sequential()
netD:add(nn.Linear(opt.n[1], HUs))
netD:add(nn.Add(HUs, false))
netD:add(nn.Linear(HUs, 1))
netD:add(nn.Sigmoid())

local data = torch.Tensor(100, opt.n[1]) 
for i = 1, data:size(1) do
    x = math.random(1000)
    data[i] =  torch.Tensor(opt.n[1])
    for j = 1, opt.n[1] do
        data[i][j] = x + ( (j-1) * 10 )
    end
end
local dataset = tensor_to_dataset(data)

function callback(epoch)
    if epoch % 10 == 0 then
        local n = torch.Tensor(20, opt.nz[1]);
        if opt.noise == 'uniform' then -- regenerate random noise
            n:uniform(-1, 1)
        elseif opt.noise == 'normal' then
            n:normal(0, 1)
        end
        if opt.gpu > 0 then
            n = n:cuda();
        end

        local r = netG:forward(n);

        print(r)
    end
end

train(netG, netD, dataset, opt, callback)
callback(0)
