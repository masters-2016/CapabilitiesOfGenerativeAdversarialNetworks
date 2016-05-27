require '../lib/generation'

local checkpoint_dir = 'checkpoints'
local gen_dir = 'gen'

paths.mkdir(gen_dir)

for f in paths.files(checkpoint_dir, function(nm) return nm:find('_G.t7') end) do
    name, epochs = table.unpack( string.split(f, "_") )

    print("Generating '" .. name .. "_" .. epochs .. "' ...")
    generate_image(checkpoint_dir .. '/' .. f, 10, 24, 100, {}, gen_dir .. '/' ..name .. '_' .. epochs .. '.png')
end
