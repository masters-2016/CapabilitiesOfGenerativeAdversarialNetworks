require 'torch'
require 'audio'

local scale_factor = math.pow(2, 31)

function load_audio(file)
    local signal, _ = audio.load(file)
    local result = torch.Tensor(signal:size(1))
    for i = 1, signal:size(1) do
        result[i] = signal[i][1] / scale_factor
    end
    return result
end

function save_audio(tensor, file)
    local a = torch.Tensor(tensor:size(1), 1)
    for i = 1, tensor:size(1) do
        a[i][1] = tensor[i] * scale_factor
    end
    audio.save(file, a, 44100);
end

function filter_audio(a, kern_size, filter_fun)
    local side_size = math.floor(kern_size / 2.0)

    local result = torch.Tensor(a:size(1))
    for i = 1, a:size(1) do
        local aux = torch.Tensor(kern_size)

        local c = 1
        for j = -side_size+1, side_size do
            local idx = i - j
            if idx < 1 or idx > a:size(1) then
                aux[c] = 0
            else
                aux[c] = a[idx]
            end
            c = c + 1
        end

        result[i] = filter_fun(aux)
    end
    return result
end

function average_filter_fun(kern)
    local sum = 0.0
    for i = 1, kern:size(1) do
        sum = sum + kern[i]
    end
    return sum / kern:size(1)
end

function median_filter_fun(kern)
    local values = {}
    for i = 1, kern:size(1) do
        table.insert(values, kern[i])
    end
    table.sort(values)

    if #values % 2 == 0 then
        return (values[ math.floor(#values / 2.0) ] + values[ math.ceil(#values / 2.0) ]) / 2.0
    else
        return values[math.floor(#values / 2)]
    end
end

function normalize_audio(a)
    local max = -9.9
    for i = 1, a:size(1) do
        max = math.max(max, math.abs(a[i]))
    end
    local scale = 0.9 / max

    local result = torch.Tensor(a:size(1))
    for i = 1, a:size(1) do
        result[i] = a[i] * scale
    end
    return result
end

function crop_audio_dir(input_dir, output_dir, length, overlap)
    paths.mkdir(output_dir)

    local count = 1
    for f in paths.files(input_dir, function(nm) return nm:find('.wav') or nm:find('.mp3') end) do
        -- Define the input file path
        local in_file = paths.concat(input_dir, f)

        -- Load the image
        local a = load_audio(in_file)
        local alen = a:size(1)

        local i = 1
        while i <= alen - length do
            -- Define the output file path
            local out_file = paths.concat(output_dir, string.format("%06d.wav", count))

            -- Produce the audio snippet
            local o = torch.Tensor(length)
            for j = 1, length do
                o[j] = a[i + (j-1)]
            end

            -- Save the audio file
            save_audio(o, out_file)

            -- Increment the counters
            i = i + (length - overlap)
            count = count + 1
        end
    end
end
