require 'image'

function load_image(filename)
    return image.load(filename)
end

function save_image(filename, img)
    image.save(filename, img)
end

function save_images(filename, imgs)
    image.save(filename, image.toDisplayTensor(imgs))
end
