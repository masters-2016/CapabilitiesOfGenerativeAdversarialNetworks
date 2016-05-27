require '../lib/generation'

if #arg ~= 4 then
	print("Usage: th generate.lua <checkpointDir> <experimentName> <outputDir> <labels (comma separated)>")
	return
end

local checkpointDir = arg[1]
local experimentName = arg[2]
local outputDir = arg[3]

local labelStrings = string.split(arg[4], ',')
local labels = {}
for i = 1, #labelStrings do
	table.insert(labels, tonumber(labelStrings[i]))
end

local noiseCount = 100 - #labels

paths.mkdir(outputDir)

-- http://stackoverflow.com/questions/24821045/does-lua-have-something-like-pythons-slice
function table.slice(tbl, first, last, step)
  local sliced = {}

  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end

  return sliced
end

for f in paths.files(checkpointDir, function(nm) return nm:find('_G.t7') end) do
    ss = string.split(f, "_")
    name = table.concat(table.slice(ss, 1, #ss - 3), "_")
    epochs = ss[#ss - 2] 
    print(name)
    print(epoch)

    if name == experimentName then
		generate_image(
			checkpointDir .. '/' .. f, 
			10, 24, noiseCount, labels, 
			outputDir .. '/' ..name .. '_' .. epochs .. '_' .. table.concat(labelStrings, "_") .. '.png')
	end
end
