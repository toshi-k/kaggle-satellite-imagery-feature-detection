
------------------------------
-- library
------------------------------

require 'nn'
require 'cunn'

------------------------------
-- setting
------------------------------

cmd = torch.CmdLine()
cmd:text('Options:')
-- global:
cmd:option('-seed', 0, 'fixed input seed for repeatable experiments')
cmd:option('-class', 1, 'target class (1-10)')
-- path:
cmd:option('-path_models', '_models', 'subdirectory to save models')
cmd:option('-path_result', 'submission_pre', 'subdirectory to result file')
-- training:
cmd:option('-learningRate', 1e-4, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-batchSize', 8, 'mini-batch size')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

seed = opt.seed
math.randomseed(seed)
torch.manualSeed(seed)
cutorch.manualSeed(seed)

------------------------------
-- main
------------------------------

-- display parameters --------

print(string.format("\tseed: %d class: %d", seed, opt.class))

-- start timer ---------------

timer = torch.Timer()

-- load files ----------------

dofile "1_data.lua"
dofile "2_model.lua"
dofile "3_train.lua"
dofile "4_test.lua"
dofile "5_valid.lua"

-- train ---------------------

for Itr = 1,5 do
	train()
	valid()
end

-- test ----------------------

test()

-- display elapsed time ------

minute = timer:time().real / 60.0
print(string.format("Elapsed Time: %.1f [min]", minute))
