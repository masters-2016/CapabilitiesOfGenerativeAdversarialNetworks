require '../lib/audio_utils'

input_dir = arg[1]
output_dir = arg[2]
length = tonumber( arg[3] )
overlap = tonumber( arg[4] )

print("Cropping audio to length " .. length .. " with overlap " .. overlap .. " from '" .. input_dir .. "' to '" .. output_dir .."'")

crop_audio_dir(input_dir, output_dir, length, overlap)
