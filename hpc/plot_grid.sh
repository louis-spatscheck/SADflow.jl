#!/bin/bash

kappas=(0.2485 0.2 0.15) 
lambdas=(0.005 0.05 0.5)
Ns=6000
nfcgs=2000000
Nevals=120000
epochs_loads="200:200:21000"
epochs_shows="600,2000,4000,6000"
snrt=5

for N in "${Ns[@]}"
do
	for nfcg in "${nfcgs[@]}"
	do
		for Neval in "${Nevals[@]}"
		do
			for epochs_load in "${epochs_loads[@]}"
			do
				for epochs_show in "${epochs_shows[@]}"
				do
					for kappa in "${kappas[@]}"
					do
						for lambda in "${lambdas[@]}"
						do
							sbatch plot.job $kappa $lambda $N $nfcg $Neval $epochs_load $epochs_show $snrt							
						done
					done
				done
			done
		done
	done
done
