#!/bin/bash
# "Cora", "CiteSeer", "PubMed","Computers"."Photo"."CS","Physics"
template='singularity exec --nv ../SIF/bgpmv116.sif python3 ./run_model.py --task GCL --model DGI --dataset Computers --config_file random_config/'

for ((i=0; i<20; i++)); do{
    config_file="config_${i}"
    command="${template}${config_file}"
    eval "$command" 2>&1 >/dev/null #21:20
    } &
done
wait