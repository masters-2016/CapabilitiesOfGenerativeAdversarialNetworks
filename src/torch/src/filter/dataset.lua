require 'torch'

require 'utils.lua'

function load_dataset(dirname, count, file_extension)
    local x = torch.Tensor(count, 3, 64, 64)
    local y = torch.Tensor(count, 3, 64, 64)

    for i = 1, count do
        x_filename = ('%s/%06d_in.%s'):format(dirname, i, file_extension)
        y_filename = ('%s/%06d_out.%s'):format(dirname, i, file_extension)

        x[i] = load_image(x_filename)
        y[i] = load_image(y_filename)
    end

    return x, y
end
