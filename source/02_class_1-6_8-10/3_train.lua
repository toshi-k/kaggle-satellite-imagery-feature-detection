
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'cunn'

------------------------------
-- main
------------------------------

-- Retrieve parameters and gradients:
if model then
	parameters,gradParameters = model:getParameters()
end

-- optimizer
print '==> configuring optimizer'
optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
}
optimMethod = optim.adam

------------------------------
-- function
------------------------------

function train()

	-- number of data
	local train_size = train_data_three:size(1)

	-- epoch tracker
	epoch = epoch or 1

	-- set model to training mode
	model:training()

	-- shuffle at each epoch
	local shuffle = torch.randperm(train_size)
	local bce_loss = 0

	-- do one epoch
	print(sys.COLORS.cyan .. '==> training on train set: # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,train_size,opt.batchSize do

		-- disp progress
		xlua.progress(t, train_size)

		local local_batchSize = math.min(opt.batchSize, train_size - t + 1)

		-- create mini batch
		local inputs_three_size = train_data_three:size()
		inputs_three_size[1] = local_batchSize
		local inputs_three = torch.Tensor(inputs_three_size)

		local inputs_sixteen_size = train_data_sixteen:size()
		inputs_sixteen_size[1] = local_batchSize
		local inputs_sixteen = torch.Tensor(inputs_sixteen_size)

		local target_size = train_cleaned_data:size()
		target_size[1] = local_batchSize
		local targets = torch.Tensor(target_size)

		for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do
			-- load new sample
			inputs_three[{{local_count}}] = train_data_three[shuffle[i]]
			inputs_sixteen[{{local_count}}] = train_data_sixteen[shuffle[i]]
			targets[{{local_count}}] = train_cleaned_data[shuffle[i]]
		end

		inputs_three = inputs_three:cuda()
		inputs_sixteen = inputs_sixteen:cuda()
		targets = targets:cuda()

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- estimate output
			local output = model:forward{inputs_three, inputs_sixteen}

			-- f is the average of all criterions
			local f = criterion:forward(output, targets)
			bce_loss = bce_loss + f * local_batchSize

			-- estimate df/dW
			local df_do = criterion:backward(output, targets)
			model:backward({inputs_three, inputs_sixteen}, df_do)

			-- normalize gradients
			gradParameters:div(local_batchSize)

			-- return f and df/dX
			return f,gradParameters
		end

		-- optimize on current mini-batch
		optimMethod(feval, parameters, optimState)
	end
	xlua.progress(train_size, train_size)

	-- calc train score
	train_score = math.sqrt(bce_loss / train_size)
	print("\ttrain_score: " .. string.format("%.5f", train_score))

	-- save/log current net
	local file_name = 'model_epoch' .. epoch .. '.net'
	local file_path = paths.concat(opt.path_models, file_name)

	paths.mkdir(sys.dirname(file_path))
	-- print('\tsaving model to '..file_path)
	torch.save(file_path, model:clearState())

	-- next epoch
	epoch = epoch + 1
end
