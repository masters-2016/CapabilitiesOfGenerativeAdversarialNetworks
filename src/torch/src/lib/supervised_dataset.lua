require 'torch'
require 'image'
require 'lfs'

require '../lib/audio_utils'

function get_filenames_and_labels(filename)
    local data = {}

    for line in io.lines(filename) do
        local filename, label_string = table.unpack( string.split(line, ' ') )
        local label_strings = string.split(label_string, ',')

        local labels = {}
        for i = 1, #label_strings do
            table.insert(labels, tonumber(label_strings[i]))
        end

        local entry = {}
        entry.filename = filename
        entry.labels = labels

        table.insert(data, entry)
    end

    return data
end

function load_supervised_image_dataset_in_memory(dirname, filename)
    local filenames_and_labels = get_filenames_and_labels(dirname .. '/' .. filename)

    local images = {}
    for i = 1, #filenames_and_labels do
        local img = image.load(dirname .. '/' .. filenames_and_labels[i].filename)

        local entry = {}
        entry.data = img

		local number_of_labels = #(filenames_and_labels[1].labels)

        entry.labels = torch.Tensor(number_of_labels)
		for j=1,number_of_labels do
			entry.labels[j] = filenames_and_labels[i].labels[j]
        end

        entry[1] = entry.data
        entry[2] = entry.label

        table.insert(images, entry)
    end

    function images.data_dim()
        return images[1].data:size()
    end

    function images.random_batch(batch_size)
        collectgarbage()

        local dim = images.data_dim()

        local data = torch.Tensor(batch_size, dim[1], dim[2], dim[3])
        local labels = torch.Tensor(batch_size, images[1].labels:size(1))

        for i = 1, batch_size do
            -- Select and load a random image
            local entry = images[ math.random(1, #images) ]
            data[i] = entry.data
            labels[i] = entry.labels
        end

        local batch = {}
        batch.data = data
        batch.labels = labels

        return batch
    end

    function images.count()
        return #images
    end

    function images.size()
        return #images
    end

    return images
end

function load_supervised_image_dataset(dirname, filename)
    local filenames_and_labels = get_filenames_and_labels(dirname .. '/' .. filename)

    local images = {}
    for i = 1, #filenames_and_labels do
        local img_file = filenames_and_labels[i].filename

        local entry = {}
        entry.filename = dirname .. '/' .. img_file

		local number_of_labels = #(filenames_and_labels[1].labels)

        entry.labels = torch.Tensor(number_of_labels)
		for j=1,number_of_labels do
			entry.labels[j] = filenames_and_labels[i].labels[j]
        end

        entry[1] = entry.filename
        entry[2] = entry.labels

        table.insert(images, entry)
    end

    function images.data_dim()
        return image.load(images[1].filename):size()
    end

    function images.random_batch(batch_size)
        collectgarbage()

        local dim = images.data_dim()

        local data = torch.Tensor(batch_size, dim[1], dim[2], dim[3])
        local labels = torch.Tensor(batch_size, images[1].labels:size(1))

        for i = 1, batch_size do
            -- Select and load a random image
            local entry = images[ math.random(1, #images) ]
            data[i] = image.load(entry.filename)
            labels[i] = entry.labels
        end

        local batch = {}
        batch.data = data
        batch.labels = labels

        return batch
    end

    function images.count()
        return #images
    end

    function images.size()
        return #images
    end

    return images
end
