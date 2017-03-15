
------------------------------
-- function
------------------------------

function getFilename( folder )

	local t = {}
	local filenames = {}

	local handle = io.popen("ls -l "..folder)
	local result = handle:read("*a")
	handle:close()

	ls_table = string.split(result, "\n")
	for i,str in ipairs(ls_table) do
		if i > 1 then
			t = string.split( str, " " )
			table.insert(filenames, t[#t])
		end
	end

	return filenames  
end
