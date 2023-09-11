#! /bin/bash
singularity exec --writable-tmpfs --nv --nvccli ../SIF/PyG.sif python3 ./run_model.py --task GCL --model DGI --dataset Planetoid --config_file config1