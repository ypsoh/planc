#!/bin/bash

TENSOR_DIR=~/hpctensor/data
#RUN_LIST=("nips" "uber" "chicago" "vast" "enron" "nell" "flickr" "deli")
RUN_LIST=("nell-2")
#RUN_LIST=("flickr")
BASE_COMMAND="./sparse_ntf"
#RANKS="16 32 40 64 128"
RANKS="32"
OFFLOAD_TYPE="1"
#~/hpctensor/data/flickr-4d.tns -r 64 -t 4 --con=nonneg"
# ./sparse_ntf -i ~/hpctensor/data/flickr-4d.tns -k 64 --gpu_offload 2 -a 4 -t 10 -e 1

UPDATE_TYPE="4"

export OMP_NUM_THREADS=26
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Command: $BASE_COMMAND"
echo

for full_path in "$TENSOR_DIR"/*.tns; do
	#filename=$(basename "$full_path")
	file_name_with_ext="${full_path##*/}"
	file_name="${file_name_with_ext%.tns}"

	#echo $filename

	for keyword in "${RUN_LIST[@]}";
	do
		if [[ "$file_name" == *"$keyword"* ]]; then
			for update_type in $UPDATE_TYPE;
			do
				for d in $RANKS;
				do
					for type in $OFFLOAD_TYPE;
					do
						logfile="__log_${file_name}_offload_${type}_rank_${d}_upd_${update_type}.log"
						echo "$logfile"
						output=$($BASE_COMMAND -i $full_path -k $d --gpu_offload $type -a $update_type -t 10 -e 1 > >(tee "$logfile") 2>&1)
						$(mv nvblas.log nvblas.$file_name.$update_type.log)
						#echo "$output"
#						echo "-> $output"
					done
				done
			done
			break
		fi
	done
done
