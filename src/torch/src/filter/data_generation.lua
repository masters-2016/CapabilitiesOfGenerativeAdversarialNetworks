require 'torch'
require 'nn'
require 'image'

require 'utils'


-- This function is from the DCGAN paper implementation
function load_net(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
   return net
end


torch.setdefaulttensortype('torch.FloatTensor')

print('Loading network ...')
local net = load_net('glasses_25000_net_G.t7', 0)
net:float()

print('Defining generation function ...')
function generate_dataset(output_dir, count)
    paths.mkdir(output_dir)

    local noise = torch.Tensor(1, 100, 1, 1)
    local image = torch.Tensor(3, 64, 64)

    for i = 1, count do
        noise:uniform(-1, 1)

        -- Generate without glasses
        noise[1][1][1][1] = -1
        image = net:forward(noise)[1]
        filename = ('%s/%06d_in.png'):format(output_dir, i)
        save_image(filename, image)

        -- Generate with glasses
        noise[1][1][1][1] = 1
        image = net:forward(noise)[1]
        filename = ('%s/%06d_out.png'):format(output_dir, i)
        save_image(filename, image)

        print(('\t%d / %d'):format(i, count))
    end
end

print('Generating training dataset ...')
generate_dataset('training_dataset', 10000)

print('Generating test dataset ...')
generate_dataset('test_dataset', 100)
