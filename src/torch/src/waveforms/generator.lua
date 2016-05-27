require 'torch'
require '../lib/generation'
require "../lib/audio_utils.lua"
util = paths.dofile('../lib/util.lua')

if #arg ~= 2 then
	print("Usage: th generate.lua <inputNet> <outputDir>")
	return
end

local inputNet = arg[1]
local outputDir = arg[2]

paths.mkdir(outputDir)


torch.setdefaulttensortype('torch.FloatTensor')

local netG = util.load(inputNet, 0)
netG:float()

local n = torch.Tensor(1024, 30, 1, 1)
n:uniform(-1, 1)
local r = netG:forward(n);
for i = 1, 1024 do
    local signal = torch.Tensor(64)
    for j = 1, 64 do
        -- MAKES TOTAL SENS
        signal[j] = r[i][1][j][1]
    end

    save_audio(signal, ("%s/%06d.wav"):format(outputDir, i))
end
