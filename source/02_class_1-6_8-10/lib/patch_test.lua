
------------------------------
-- library
------------------------------

require 'gfx.js'
require 'patch'
require 'image'
require 'window'

------------------------------
-- main
------------------------------

img = image.load('lena512.png')
-- gfx.image(img)
print(img:size())

for i = 1,100 do

	local patch_size = math.random(512)
	local overlap = math.random(patch_size-1)
	local move_over_x = math.random(patch_size-1)
	local move_over_y = math.random(patch_size-1)
	-- local move_over = 1

	local patch_param = {patch_size, overlap, move_over_x, move_over_y}

	print('patch: ' .. tostring(patch_param[1]) ..
		' overlap: ' .. tostring(patch_param[2]) ..
		' move_x: ' .. tostring(patch_param[3]) ..
		' move_y: ' .. tostring(patch_param[4]))

	local data = img2patch({img}, patch_param)
	-- print(data:size())

	local img2 = patch2img(data, {img}, patch_param)[1]

	-- gfx.image(data)
	-- gfx.image(img2)

	local diff = torch.sum(img - img2)
	assert(diff < 1e-10)
end
