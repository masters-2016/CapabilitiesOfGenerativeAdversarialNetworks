require 'torch'
require 'nn'
require 'audio'

require "../lib/audio_utils.lua"
require "../lib/dataset.lua"

-- Get command line args
function usage_and_exit()
    print("Computes the internal and external similarities of a generated sounds")
    print("")
    print("Usage: th compute_similarities.lua <input_audio_dir> <input_dataset_dir>")
    print("")
    print("Implemented algorithms:")
    print("- 'l2': l2 / euclidean distance")

    os.exit()
end

if #arg ~= 2 then
    usage_and_exit()
end

local input_audio_dir = arg[1]
local input_dataset_dir = arg[2]

function l2_similarity(image,images)
    local count = 0
    if images.count then
        count = images.count()
    else
        count = images:size(1)
    end

    local result = torch.Tensor(count)

    for i = 1, count do
        local diffs = torch.pow(image - images[i],2)
        local sum = torch.sum(diffs)

        result[i] = math.sqrt(sum)
    end

    return result
end

-- Define the similarity and conversion functions
similarity_func = l2_similarity

-- Load input image
print('Loading generated audio...')
local input_audio = load_audio_dir_in_memory(input_audio_dir)


-- Load the dataset
print('Loading dataset audio...')
local dataset = load_audio_dir_in_memory(input_dataset_dir)


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

for i = 1, input_audio.count()-1 do
    local count = 0

    -- Compute the similarities
    local audio = input_audio[i]
    local audios = torch.Tensor(input_audio.count()-i, audio:size(1))

    for j = i+1, input_audio.count() do
        audios[j-i] = input_audio[j]
    end

    local similarities = similarity_func(audio, audios)

    -- Assign the similarities
    for j = i+1, input_audio.count() do
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

for i = 1, input_audio.count() do
    -- Compute the similarities
    local audio = input_audio[i]

    local similarities = similarity_func(audio, dataset)

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
    input_audio.count(),
    input_audio.count())

pretty_print(
    "External similarity",
    min_external_similarity,
    mean_external_similarity,
    max_external_similarity,
    input_audio.count(),
    dataset.count())
