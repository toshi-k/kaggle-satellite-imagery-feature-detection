
------------------------------
-- function
------------------------------

function hanning(size)
	local v = torch.range(0,size-2)/(size-1)
	local v2 = torch.Tensor(v:size(1)+1):fill(1)

	v2[{{1,v:size(1)}}] = v
	hanv = - torch.cos(v2 * 2 * math.pi)*0.5 + 0.5
	ret = torch.Tensor(v2:size(1),v2:size(1)):zero()
	ret:addr(hanv,hanv):add(0.01)
	ret = ret/torch.sum(ret)

	return ret
end
