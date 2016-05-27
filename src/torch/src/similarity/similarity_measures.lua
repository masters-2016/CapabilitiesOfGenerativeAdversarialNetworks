require 'torch'
require 'nn'

local net = torch.load('net.t7')

function cvpr_similarity(image, images)
    -- Makes use of: https://github.com/szagoruyko/cvpr15deepcompare.git
    -- The images must be grayscale

    local count = images:size(1)

    -- Make the tensor of image patches to compare
    local patches = torch.Tensor(count, 2, 64, 64):float()
    for i = 1, count do
        patches[i][1] = image[1]
        patches[i][2] = images[i][1]
    end

    -- Apply the similarity network
    local p = patches:view(patches:size(1), 2, 64 * 64)
    p:add(-p:mean(3):expandAs(p))

    local similarities = net:forward(patches)

    local result = torch.Tensor(count)
    for i = 1, count do
        result[i] = similarities[i][1]
    end

    return result
end

function random_similarity(image, images)
    return torch.Tensor(images:size(1)):uniform(0, 1)
end

function pixel_similarity(image, images)
    local count = images:size(1)

    local result = torch.Tensor(count)

    for i = 1, count do
        local diffs = torch.abs( image - images[i] )
        result[i] = torch.mean(diffs)
    end

    return result
end

function l2_similarity(image,images)
    local count = images:size(1)

    local result = torch.Tensor(count)

    for i = 1, count do
        local diffs = torch.pow(image - images[i],2)
        local sum = torch.sum(diffs)

        result[i] = math.sqrt(sum)
    end

    return result
end
