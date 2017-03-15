
------------------------------
-- library
------------------------------

require 'torch'
require 'image'

-- require 'gfx.js'

require "lib/window"
require "lib/patch"
require "lib/vgg_preprocess"

npy4th = require 'npy4th'

------------------------------
-- function
------------------------------

function load_imgs(folder_path, is_preprocess, npy)

	local images = {}

	local load_f
	if npy == true then
		load_f = npy4th.loadnpy
	else
		load_f = image.load
	end

	for file_name in paths.iterfiles(folder_path) do
		local file_path = paths.concat(folder_path, file_name)
		if is_preprocess == true then
			table.insert(images, preprocess(load_f(file_path)))
		else
			table.insert(images, load_f(file_path))
		end
	end

	return images
end

------------------------------
-- main
------------------------------

-- patch_size, overlap, move_over_x, move_over_y
patch_param = {224, 16, math.random(224-1), math.random(224-1)}

print('\tpatch: ' .. tostring(patch_param[1]) ..
	' overlap: ' .. tostring(patch_param[2]) ..
	' move_x: ' .. tostring(patch_param[3]) ..
	' move_y: ' .. tostring(patch_param[4]))

train_images_three = load_imgs("../../input/train_input_three", true, false)
train_images_sixteen = load_imgs("../../input/train_input_sixteen", false, true)
train_cleaned_images = load_imgs("../../input/train_output_class" .. opt.class, false, false)

print("\tnumber of train images: " .. #train_images_three)
assert(#train_images_three == #train_images_sixteen)
assert(#train_images_three == #train_cleaned_images)

-- gfx.image(train_images[1], 'data 1')
-- gfx.image(train_images_sixteen[1][{{1,3}}], 'data 1')

-- gfx.image(train_images[2], 'data 2')
-- gfx.image(train_images_sixteen[2][{{1,3}}], 'data 2')

train_data_three = img2patch(train_images_three, patch_param)
train_data_sixteen = img2patch_scale(train_images_sixteen, train_images_three, patch_param)
train_cleaned_data = img2patch(train_cleaned_images, patch_param)

-- gfx.image(train_data[1], 'data 1')
-- gfx.image(train_data_sixteen[1][{{1,3}}], 'data 1')

-- gfx.image(train_data[2], 'data 2')
-- gfx.image(train_data_sixteen[2][{{1,3}}], 'data 2')

valid_rate = 0.1
num_valid = math.ceil(train_data_three:size(1) * valid_rate)
print(string.format("\tnum of train patch: %d", train_data_three:size(1) - num_valid))
print(string.format("\tnum of valid patch: %d", num_valid))

split_index = torch.randperm(train_data_three:size(1)):long()

valid_index = split_index[{{1, num_valid}}]
train_index = split_index[{{num_valid+1, split_index:size(1)}}]

valid_data_three = train_data_three:index(1, valid_index)
valid_data_sixteen = train_data_sixteen:index(1, valid_index)
valid_cleaned_data = train_cleaned_data:index(1, valid_index)

train_data_three = train_data_three:index(1, train_index)
train_data_sixteen = train_data_sixteen:index(1, train_index)
train_cleaned_data = train_cleaned_data:index(1, train_index)

-- check that both train and valid have positive example
assert(train_cleaned_data:sum() > 0)
assert(valid_cleaned_data:sum() > 0)
