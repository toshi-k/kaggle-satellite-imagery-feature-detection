#!/bin/sh

for seed in `seq 1 10`
do
	for i in 1 2 3 4 5 6 8 9 10
	do
		echo "run with class: ${i}"
		th main.lua -class ${i} -seed ${seed}
	done
done
