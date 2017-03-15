
------------------------------
-- function
------------------------------

-- img2patch
function img2patch(images, patch_param)

	local patch_size, overlap, move_over_x, move_over_y = table.unpack(patch_param)

	local num_channel = images[1]:size(1)

	local patch_num = 0
	local step = patch_size - overlap

	for i = 1,#images do
		local num_y = math.ceil((images[i]:size(2) + move_over_y - patch_size) / step) + 1
		local num_x = math.ceil((images[i]:size(3) + move_over_x - patch_size) / step) + 1
		patch_num = patch_num + num_x * num_y
	end

	local count = 1
	local data = torch.Tensor(patch_num, num_channel, patch_size, patch_size)

	for i = 1,#images do

		local size_x = images[i]:size(3)
		local size_y = images[i]:size(2)
		local num_y = math.ceil((size_y + move_over_y - patch_size) / step) + 1
		local num_x = math.ceil((size_x + move_over_x - patch_size) / step) + 1

		for sx = 1,num_x do
			for sy = 1,num_y do
				local x = 1 + (patch_size - overlap) * (sx - 1)
				local y = 1 + (patch_size - overlap) * (sy - 1)

				x = math.max(x - move_over_x, 1)
				y = math.max(y - move_over_y, 1)

				x = math.min(x, size_x - patch_size + 1)
				y = math.min(y, size_y - patch_size + 1)

				img = images[i]
				data[{{count}}] = img[{{}, {y,y+patch_size-1},{x,x+patch_size-1}}]
				count = count + 1
			end
		end
	end

	return data
end

-- img2patch
function img2patch_scale(images, ref, patch_param)

	local patch_size, overlap, move_over_x, move_over_y = table.unpack(patch_param)

	local num_channel = images[1]:size(1)

	local patch_num = 0
	local step = patch_size - overlap

	for i = 1,#ref do
		local num_y = math.ceil((ref[i]:size(2) + move_over_y - patch_size) / step) + 1
		local num_x = math.ceil((ref[i]:size(3) + move_over_x - patch_size) / step) + 1
		patch_num = patch_num + num_x * num_y
	end

	local count = 1
	local data = torch.Tensor(patch_num, num_channel, patch_size/4, patch_size/4)

	for i = 1,#ref do

		local size_x = ref[i]:size(3)
		local size_y = ref[i]:size(2)
		local num_y = math.ceil((size_y + move_over_y - patch_size) / step) + 1
		local num_x = math.ceil((size_x + move_over_x - patch_size) / step) + 1

		local img_large = image.scale(images[i], size_x, size_y, 'simple')

		for sx = 1,num_x do
			for sy = 1,num_y do
				local x = 1 + (patch_size - overlap) * (sx-1)
				local y = 1 + (patch_size - overlap) * (sy-1)

				x = math.max(x - move_over_x, 1)
				y = math.max(y - move_over_y, 1)

				x = math.min(x, size_x - patch_size + 1)
				y = math.min(y, size_y - patch_size + 1)

				local img_crop = img_large[{{}, {y,y+patch_size-1},{x,x+patch_size-1}}]
				local img_small = image.scale(img_crop, patch_size/4, patch_size/4, 'simple')
				data[{{count}}] = img_small
				count = count + 1
			end
		end
	end

	return data
end

-- patch2img
function patch2img(data, original_images, patch_param)

	local patch_size, overlap, move_over_x, move_over_y = table.unpack(patch_param)
	local num_channel = original_images[1]:size(1)

	local count = 1
	local images = {}
	local window = hanning(patch_size)
	window = window:repeatTensor(num_channel,1,1)

	local step = patch_size - overlap

	for i = 1,#original_images do
		local size_x = original_images[i]:size(3)
		local size_y = original_images[i]:size(2)

		images[i] = torch.Tensor(num_channel, size_y, size_x):zero()
		local weight = torch.Tensor(num_channel, size_y, size_x):zero()
		local num_y = math.ceil((size_y + move_over_y - patch_size) / step) + 1
		local num_x = math.ceil((size_x + move_over_x - patch_size) / step) + 1

		local img_copy = images[i]

		for sx = 1,num_x do
			for sy = 1,num_y do
				local x = 1 + (patch_size - overlap) * (sx - 1)
				local y = 1 + (patch_size - overlap) * (sy - 1)

				x = math.max(x - move_over_x, 1)
				y = math.max(y - move_over_y, 1)

				if x + patch_size-1 > size_x then x = size_x - patch_size + 1 end
				if y + patch_size-1 > size_y then y = size_y - patch_size + 1 end

				local add_data = torch.cmul(data[{count}], window)

				img_copy[{{},{y,y+patch_size-1},{x,x+patch_size-1}}]:add(add_data)
				weight[{{},{y,y+patch_size-1},{x,x+patch_size-1}}]:add(window)

				count = count + 1
			end
		end

		images[i]:cdiv(weight)
	end

	return images
end
