
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'

require "lib/getfilename"
require "lib/patch"
require "lib/vgg_preprocess"

------------------------------
-- function
------------------------------

function test()

	-- set model to evaluate mode
	model:evaluate()

	local test_three_dir = "../../input/test_input_three"
	local test_sixteen_dir = "../../input/test_input_sixteen"

	local test_files = getFilename(test_three_dir)

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
	for t, test_file in ipairs(test_files) do

		-- disp progress
		xlua.progress(t, #test_files)

		-- file pat for test image
		local test_three_img_path = paths.concat(test_three_dir, test_file)
		local test_sixteen_img_path = paths.concat(test_sixteen_dir, string.sub(test_file,0,-4) .. "npy")

		-- load test image
		local test_img = preprocess(image.load(test_three_img_path))
		local test_sixteen_img = npy4th.loadnpy(test_sixteen_img_path)

		local test_result_size = test_img:size()
		local test_result_dummy = torch.Tensor(1, test_img:size(2), test_img:size(3))

		local test_three_data = img2patch({test_img}, patch_param)
		local test_data_sixteen = img2patch_scale({test_sixteen_img}, {test_img}, patch_param)

		local test_data_pred_size = test_three_data:size()
		test_data_pred_size[2] = 1
		local test_data_pred = torch.Tensor(test_data_pred_size)

		for i = 1,test_three_data:size(1) do

			local input_three = test_three_data[{{i}}]
			local input_sixteen = test_data_sixteen[{{i}}]

			input_three = input_three:cuda()
			input_sixteen = input_sixteen:cuda()

			local output = model:forward{input_three, input_sixteen}
			test_data_pred[{{i}}] = output:float()
		end

		local test_img_pred = patch2img(test_data_pred, {test_result_dummy}, patch_param)[1]
		-- test_img_pred = test_img_pred:ge(0.1)
		test_img_pred = test_img_pred:float()

		local output_dir = paths.concat("../../submission/", opt.path_result,
			string.format("class%d", opt.class) ..
			string.format("_valid%.3f", valid_jaccard) ..
			string.format("_threshold%.3f", threshold) ..
			string.format("_seed%d", seed))

		paths.mkdir(output_dir)
		image.save(paths.concat(output_dir, test_file), test_img_pred)
	end
	xlua.progress(#test_files, #test_files)

end
