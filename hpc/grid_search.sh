#!/bin/bash

kappas=(0.2485 0.2 0.15) 
lambdas=(0.0 0.005 0.05 0.5)
epochs=100000
batchsizes=(256)
depths=1
nodes=4
learning_rates=(0.001 0.01)
modes=(1 2 5)
activations=("relu")

for batchsize in "${batchsizes[@]}"
do
	for depth in "${depths[@]}"
	do
		for mode in "${modes[@]}"
		do
			for learning_rate in "${learning_rates[@]}"
			do
				for activation in "${activations[@]}"
				do
					for kappa in "${kappas[@]}"
					do
						for lambda in "${lambdas[@]}"
						do
							sbatch train.job $kappa $lambda $epochs $batchsize $depth $nodes $learning_rate $mode $activation
						done
					done
				done
			done
		done
	done
done
