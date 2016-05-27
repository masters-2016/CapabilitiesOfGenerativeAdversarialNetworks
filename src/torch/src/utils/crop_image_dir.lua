require '../lib/image_utils'

input_dir = arg[1]
output_dir = arg[2]
dim = tonumber( arg[3] )

print("Cropping images to size " .. dim .. "x" .. dim .. " from '" .. input_dir .. "' to '" .. output_dir .."'")

crop_image_dir(input_dir, output_dir, dim)
