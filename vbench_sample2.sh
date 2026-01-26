export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 --master_port=28500 sample_vbench.py --config_path configs/longlive_vbench.yaml

cd ..
cd ./VBench
bash ada_evaluate.sh /home/hangguo/dataset/20260107_single5s_longlive_teacache_ours /home/hangguo/VBench/sample_vbench/20260107_single5s_longlive_teacache_ours

