require 'image'

-- Convert the input images to grayscale
-- function from https://gist.github.com/jkrish/29ca7302e98554dd0fcb
function rgb2gray(im)
	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]

	local r = im:select(1, 1)
	local g = im:select(1, 2)
	local b = im:select(1, 3)

	local z = torch.Tensor(w, h):zero()

	z = z:add(0.21, r)
	z = z:add(0.72, g)
	z = z:add(0.07, b)

	return z
end

function rgb2rgb(im)
    return im
end

function rgb_resize(im)
    return image.scale(im, 16, 16)
end
