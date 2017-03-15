
------------------------------
-- function
------------------------------

function preprocess(img)
	img[{{1}}]:add(-123.68 / 255.0)  -- Red
	img[{{2}}]:add(-116.779 / 255.0) -- Green
	img[{{3}}]:add(-103.939 / 255.0) -- Blue
	return img
end
