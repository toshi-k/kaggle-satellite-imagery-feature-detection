
------------------------------
-- library
------------------------------

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

require 'loadcaffe'

------------------------------
-- function
------------------------------

function newmodel()

	-- filter size
	local filtsize = 3

	-- hidden units
	local nstates = {512,256,256,256,128,128,64,64}

	-- load pretarin model:
	local vgg = loadcaffe.load('pretrain/vgg16/deploy.prototxt',
		'pretrain/vgg16/VGG_ILSVRC_16_layers.caffemodel', 'cudnn')

	-- model for three band
	local model_three = nn.Sequential()

	-- stage 1 : Convolution
	for i = 1,4 do
		model_three:add(vgg:get(i))
	end

	local max_pooling1 = nn.SpatialMaxPooling(2,2,2,2)
	model_three:add(max_pooling1)

	-- stage 2 : Convolution
	for i = 6,9 do
		model_three:add(vgg:get(i))
	end

	local max_pooling2 = nn.SpatialMaxPooling(2,2,2,2)
	model_three:add(max_pooling2)

	model_three:add(nn.Narrow(2, 1, 120))
	model_three:add(nn.SpatialBatchNormalization(120))

	local model_head = nn.ParallelTable(2)
	model_head:add(model_three)
	model_head:add(nn.SpatialBatchNormalization(8))

	local model = nn.Sequential()
	model:add(model_head)
	model:add(nn.JoinTable(2))

	-- stage 3 : Convolution
	for i = 11,16 do
		model:add(vgg:get(i))
	end

	local max_pooling3 = nn.SpatialMaxPooling(2,2,2,2)
	model:add(max_pooling3)

	-- stage 4 : Convolution
	for i = 18,23 do
		model:add(vgg:get(i))
	end

	local max_pooling4 = nn.SpatialMaxPooling(2,2,2,2)
	model:add(max_pooling4)

	-- stage 5 : Convolution
	model:add(nn.SpatialMaxUnpooling(max_pooling4))

	model:add(cudnn.SpatialConvolution(nstates[1], nstates[2], filtsize, filtsize, 1, 1, 1, 1))
	model:add(nn.SpatialBatchNormalization(nstates[2]))
	model:add(nn.LeakyReLU(0.1, true))

	model:add(cudnn.SpatialConvolution(nstates[2], nstates[3], filtsize, filtsize, 1, 1, 1, 1))
	model:add(nn.SpatialBatchNormalization(nstates[3]))
	model:add(nn.LeakyReLU(0.1, true))

	-- stage 6 : Convolution
	model:add(nn.SpatialMaxUnpooling(max_pooling3))

	model:add(cudnn.SpatialConvolution(nstates[3], nstates[4], filtsize, filtsize, 1, 1, 1, 1))
	model:add(nn.SpatialBatchNormalization(nstates[4]))
	model:add(nn.LeakyReLU(0.1, true))

	model:add(cudnn.SpatialConvolution(nstates[4], nstates[5], filtsize, filtsize, 1, 1, 1, 1))
	model:add(nn.SpatialBatchNormalization(nstates[5]))
	model:add(nn.LeakyReLU(0.1, true))

	-- stage 7 : Convolution
	model:add(nn.SpatialMaxUnpooling(max_pooling2))

	model:add(cudnn.SpatialConvolution(nstates[5], nstates[6], filtsize, filtsize, 1, 1, 1, 1))
	model:add(nn.SpatialBatchNormalization(nstates[6]))
	model:add(nn.LeakyReLU(0.1, true))

	model:add(cudnn.SpatialConvolution(nstates[6], nstates[7], filtsize, filtsize, 1, 1, 1, 1))
	model:add(nn.SpatialBatchNormalization(nstates[7]))
	model:add(nn.LeakyReLU(0.1, true))

	-- stage 8 : Convolution
	model:add(nn.SpatialMaxUnpooling(max_pooling1))

	model:add(cudnn.SpatialConvolution(nstates[7], nstates[8], filtsize, filtsize, 1, 1, 1, 1))
	model:add(nn.SpatialBatchNormalization(nstates[8]))
	model:add(nn.LeakyReLU(0.1, true))

	model:add(cudnn.SpatialConvolution(nstates[8], 1, filtsize, filtsize, 1, 1, 1, 1))

	model:add(nn.Sigmoid())

	return model
end

------------------------------
-- main
------------------------------

model = newmodel()
-- print(model)

-- loss function
criterion = nn.BCECriterion()
print '==> here is the loss function:'
print(criterion)

model:cuda()
criterion:cuda()

-- samp1 = torch.Tensor(6,3,128,128):uniform()
-- samp1 = samp1:cuda()

-- samp2 = torch.Tensor(6,8,32,32):uniform()
-- samp2 = samp2:cuda()

-- output = model:forward{samp1, samp2}
-- print(output:size())
