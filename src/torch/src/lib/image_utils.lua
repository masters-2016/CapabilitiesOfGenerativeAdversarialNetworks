require 'image'

function crop_image_dir(input_dir, output_dir, dim)
    paths.mkdir(output_dir)

    for f in paths.files(input_dir, function(nm) return nm:find('.png') or nm:find('.jpg') end) do
        -- Define the file paths
        local in_file = paths.concat(input_dir, f)
        local out_file = paths.concat(output_dir, f)

        -- Load the image
        local im = image.load(in_file)
        local height, width = im:size(2), im:size(3)

        -- Scale the image so the smallest of the width and height is dim pixels
        if width > height then 
            scaled_height = dim
            scaled_width = dim * (width / height)
        else
            scaled_height = dim * (height / width)
            scaled_width = dim
        end
        local scaled = image.scale(im, scaled_width, scaled_height)

        -- Select the center dimxdim pixels
        if width > height then 
            height_border = 0
            width_border = math.floor((scaled_width - dim) / 2)
        else
            height_border = math.floor((scaled_height - dim) / 2)
            width_border = 0
        end

        x1, y1 = width_border, height_border
        x2, y2 = x1+dim, y1+dim
        local result = image.crop(scaled, x1, y1, x2, y2)

        -- Save the output image
        image.save(out_file, result)
    end
end
