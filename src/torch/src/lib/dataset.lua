require 'torch'
require 'image'
require 'lfs'

--require '../lib/audio_utils'

function load_image_dir(dirname, max_dataset_size)
    local images = {}
    local count = 0
    for file in lfs.dir(dirname) do
        if string.match(file, "jpg") or string.match(file, "png") then
            table.insert(images, dirname .. '/' ..file)
            count = count + 1

            if max_dataset_size and count >= max_dataset_size then
                break
            end
        end
    end

    function images.data_dim()
        return image.load(images[1]):size()
    end

    function images.random_batch(batch_size)
        collectgarbage()

        local dim = images.data_dim()
        local batch = torch.Tensor(batch_size, dim[1], dim[2], dim[3])
        for i = 1, batch_size do
            -- Select and load a random image
            img_path = images[ math.random(1, #images) ]
            batch[i] = image.load(img_path)
        end

        return batch
    end

    function images.count()
        return #images
    end

    return images
end

function load_image_dir_in_memory(dirname, max_dataset_size)
    local images = {}
    local count = 0
    for file in lfs.dir(dirname) do
        if string.match(file, "jpg") or string.match(file, "png") then
            img = image.load(dirname .. '/' ..file)
            table.insert(images, img)
            count = count + 1

            if max_dataset_size and count >= max_dataset_size then
                break
            end
        end
    end

    function images.data_dim()
        return images[1]:size()
    end

    function images.random_batch(batch_size)
        collectgarbage()

        local dim = images.data_dim()
        local batch = torch.Tensor(batch_size, dim[1], dim[2], dim[3])
        for i = 1, batch_size do
            -- Select and load a random image
            batch[i] = images[ math.random(1, #images) ]
        end

        return batch
    end

    function images.count()
        return #images
    end

    return images
end

function tensor_to_dataset(tensor)
    local data = {}

    function data.data_dim()
        return tensor[1]:size()
    end

    function data.random_batch(batch_size)
        local dim = {}
        local dim_storage = data.data_dim()
        for i = 1, #dim_storage do
            table.insert(dim, dim_storage[i])
        end

        local batch = torch.Tensor(batch_size, unpack(dim))
        for i = 1, batch_size do
            batch[i] = tensor[ math.random(1, tensor:size(1)) ]
        end

        return batch
    end

    function data.count()
        return tensor:size(1)
    end

    return data
end

function load_audio_dir(dirname)
    local audio_files = {}
    for file in lfs.dir(dirname) do
        if string.match(file, "wav") or string.match(file, "mp3") then
            table.insert(audio_files, dirname .. '/' .. file)
        end
    end

    function audio_files.data_dim()
        return load_audio(audio_files[1]):size()
    end

    function audio_files.random_batch(batch_size)
        collectgarbage()

        local dim = audio_files.data_dim()
        local batch = torch.Tensor(batch_size, dim[1])
        for i = 1, batch_size do
            -- Select and load a random image
            audio_path = audio_files[ math.random(1, #audio_files) ]
            batch[i] = load_audio(audio_path)
        end

        return batch
    end

    function audio_files.count()
        return #audio_files
    end

    return audio_files
end

function load_audio_dir_in_memory(dirname)
    local audio_files = {}
    for file in lfs.dir(dirname) do
        if string.match(file, "wav") or string.match(file, "mp3") then
            table.insert(audio_files, load_audio(dirname .. '/' .. file))
        end
    end

    function audio_files.data_dim()
        return audio_files[1]:size()
    end

    function audio_files.random_batch(batch_size)
        collectgarbage()

        local dim = audio_files.data_dim()
        local batch = torch.Tensor(batch_size, dim[1])
        for i = 1, batch_size do
            -- Select and load a random image
            batch[i] = audio_files[ math.random(1, #audio_files) ]
        end

        return batch
    end

    function audio_files.count()
        return #audio_files
    end

    return audio_files
end
