require 'torch'
require 'nn'
require 'image'

require '../lib/dataset'
require '../similarity/similarity_measures'
require '../similarity/conversion'

-- Get command line args
function usage_and_exit()
    print("Computes the internal and external similarities of a generated image")
    print("i.e. an image consists of many 64x64, 3 channel images. It additionally")
    print("creates as output an image where each sub-image of the input image is")
    print("replaced with the most similar image from the training data set.")
    print("")
    print("Usage: th compute_similarities.lua <dataset_dir> <input_image_file> <output_image_file> <algorithm>")
    print("")
    print("Implemented algorithms:")
    print("- 'random': No similarity is computed. It just returns a random value. Made to testing purposes.")
    print("- 'pixel': Mean pixelwise image difference.")
    print("- 'resized_pixel': Mean pixelwise image difference but with the images resized to size 16x16, 3 channels.")
    print("- 'cvpr': Network trained to compute image similarity. The network must be downloaded by running the 'download_net.sh' script. The images are converted to greyscale before any processing in done.")
    print("- 'l2': l2 / euclidean distance")

    os.exit()
end

if #arg ~= 4 then
    usage_and_exit()
end

local dataset_dir = arg[1]
local input_image_file = arg[2]
local output_image_file = arg[3]
local algorithm = arg[4]


-- Define the similarity and conversion functions
if algorithm == "cvpr" then
    similarity_func = cvpr_similarity
    conversion_func = rgb2gray
    converted_dim = {1, 64, 64}
elseif algorithm == "random" then
    similarity_func = random_similarity
    conversion_func = rgb2rgb
    converted_dim = {3, 64, 64}
elseif algorithm == "pixel" then
    similarity_func = pixel_similarity
    conversion_func = rgb2rgb
    converted_dim = {3, 64, 64}
elseif algorithm == "resized_pixel" then
    similarity_func = pixel_similarity
    conversion_func = rgb_resize
    converted_dim = {3, 16, 16}
elseif algorithm == "l2" then
    similarity_func = l2_similarity
    conversion_func = rgb2rgb
    converted_dim = {3, 64, 64}
else
    usage_and_exit()
end

-- Load input image
print('Loading input image ...')
local input_image = image.load(input_image_file)
local dim = input_image:size()

local channels = dim[1]
local height = dim[2]
local width = dim[3]


-- Crop the input image to the images of which it consists
print('Splitting input image into sub images ...')
local image_count = (width / 64) * (height / 64)
local input_images = torch.Tensor(image_count, channels, 64, 64)


local idx = 1
for h = 0, height-1, 64 do
    for w = 0, width-1, 64 do
        input_images[idx] = image.crop(input_image, w, h, w+64, h+64)
        idx = idx + 1
    end
end

image.save("test.png",image.toDisplayTensor(input_images))

print('Converting input images ...')
local converted_input_images = torch.Tensor(image_count, table.unpack(converted_dim))
for i = 1, image_count do
    converted_input_images[i] = conversion_func(input_images[i])
end


-- Load the dataset
print('Loading dataset ...')
local dataset = load_image_dir_in_memory(dataset_dir)


-- Create a grayscale dataset
print('Converting dataset ...')
local converted_dataset = torch.Tensor(#dataset, table.unpack(converted_dim))
for i = 1, #dataset do
    converted_dataset[i] = conversion_func(dataset[i])
end


-- Define the metric values
local mean_internal_similarity = 0.0
local min_internal_similarity = 99999.9
local max_internal_similarity = -99999.9

local mean_external_similarity = 0.0
local min_external_similarity = 99999.9
local max_external_similarity = -99999.9

local most_similar_indices = {}


-- Compute the internal similarity
print('Computing internal similarity ...')

local internal_sim_count = 0

for i = 1, image_count-1 do
    local count = 0

    -- Compute the similarities
    local image = converted_input_images[i]
    local images = torch.Tensor(image_count-i, image:size(1), image:size(2), image:size(3))

    for j = i+1, image_count do
        images[j-i] = converted_input_images[j]
    end

    local similarities = similarity_func(image, images)

    -- Assign the similarities
    for j = i+1, image_count do
        local sim = similarities[j-i]

        internal_sim_count = internal_sim_count + 1
        mean_internal_similarity = mean_internal_similarity + sim

        min_internal_similarity = math.min(sim, min_internal_similarity)
        max_internal_similarity = math.max(sim, max_internal_similarity)
    end
end

mean_internal_similarity = mean_internal_similarity / internal_sim_count


-- Compute the external similarity
print('Computing external similarity ...')

local external_sim_count = 0

for i = 1, image_count do
    -- Compute the similarities
    local image = converted_input_images[i]

    local similarities = similarity_func(image, converted_dataset)

    -- Find the most similar image in the dataset
    local best_idx = 0
    local min_similarity = 99999.9

    for j = 1, dataset.count() do
        local sim = similarities[j]

        mean_external_similarity = mean_external_similarity + sim
        external_sim_count = external_sim_count + 1

        min_external_similarity = math.min(sim, min_external_similarity)
        max_external_similarity = math.max(sim, max_external_similarity)

        if sim < min_similarity then
            best_idx = j
            min_similarity = sim
        end
    end

    most_similar_indices[i] = best_idx
end

mean_external_similarity = mean_external_similarity / external_sim_count


-- Collect the most similar images
print('Collecting most similar images ...')
local output_images = torch.Tensor(image_count, channels, 64, 64)
for i = 1, image_count do
    output_images[i] = dataset[most_similar_indices[i]]
end


-- Save the output image
print('Combining and saving output image ...')
local combined_image = image.toDisplayTensor(output_images)
image.save(output_image_file, combined_image)


-- Print similarity metrics
function pretty_print(title, min, mean, max, x, y)
    print("\n----- " .. title .. " -----\n")
    print("min " .. min)
    print("mean " .. mean)
    print("max " .. max)
    print("")
end

pretty_print(
    "Internal similarity",
    min_internal_similarity,
    mean_internal_similarity,
    max_internal_similarity,
    image_count,
    image_count)

pretty_print(
    "External similarity",
    min_external_similarity,
    mean_external_similarity,
    max_external_similarity,
    image_count,
    dataset.count())
