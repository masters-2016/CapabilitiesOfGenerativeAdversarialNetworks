require '../lib/generation'

if #arg ~= 3 then
	print("Usage: th generate.lua <checkpointDir> <experimentName> <outputDir>")
	return
end

local checkpointDir = arg[1]
local experimentName = arg[2]
local outputDir = arg[3]

paths.mkdir(outputDir)

for f in paths.files(checkpointDir, function(nm) return nm:find('_G.t7') end) do
    name, epochs = string.split(f, "_")

    if name == experimentName then
		generate_image(
			checkpointDir .. '/' .. f, 
			10, 24, 100, {}, 
			outputDir .. '/' ..name .. '_' .. epochs .. '.png')
	end
end
