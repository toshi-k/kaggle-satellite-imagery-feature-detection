
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'

------------------------------
-- function
------------------------------

function byte_and(tensor_a, tensor_b)
	local function f_and(xx, yy) return xx * yy end
	local ret = tensor_a:clone()
	ret:map(tensor_b, f_and)
	return ret
end

function search_best_threshold(valid_cleaned_data_pred)

	local best_score = 0
	local best_threshld = 1.0/100

	local truth = valid_cleaned_data:view(-1):ge(0.5)

	print(sys.COLORS.magenta .. '==> search best threshold:')
	for t = 1,99 do
		xlua.progress(t, 99)

		local th = t / 100
		local pred = valid_cleaned_data_pred:view(-1):ge(th)

		local tp = byte_and(truth, pred):sum()
		local fp = byte_and(1-truth, pred):sum()
		local fn = byte_and(truth, 1-pred):sum()

		-- print("tp: " .. tp .. " fp: " .. fp .. " fn: " .. fn)
		local jaccard = tp / (tp + fp + fn)

		if jaccard > best_score then
			best_score = jaccard
			best_threshld = th
		end
	end

	threshold = best_threshld
	valid_jaccard = best_score

	print("\t best_jaccard: " .. string.format("%.5f", best_score))
	print("\tbest_threshld: " .. string.format("%.5f", best_threshld))
end

function valid()

	-- set model to evaluate mode
	model:evaluate()

	local valid_cleaned_data_pred = torch.Tensor()
	valid_cleaned_data_pred:resizeAs(valid_cleaned_data)

	local valid_size = valid_data_three:size(1)
	local bce_loss = 0

	-- validate over valid data
	print(sys.COLORS.green .. '==> validating on valid set:')
	for t = 1,valid_size do

		-- disp progress
		xlua.progress(t, valid_size)

		-- get new sample
		local input_three = valid_data_three[{{t}}]
		local input_sixteen = valid_data_sixteen[{{t}}]
		local target = valid_cleaned_data[{{t}}]

		input_three = input_three:cuda()
		input_sixteen = input_sixteen:cuda()
		target = target:cuda()

		local output = model:forward{input_three, input_sixteen}
		local f = criterion:forward(output, target)
		bce_loss = bce_loss + f

		-- valid sample
		valid_cleaned_data_pred[{{t}}] = output:float()
	end
	xlua.progress(valid_size, valid_size)

	valid_score = math.sqrt(bce_loss / valid_size)
	print("\tvalid_score: " .. string.format("%.5f", valid_score))

	search_best_threshold(valid_cleaned_data_pred)
end
