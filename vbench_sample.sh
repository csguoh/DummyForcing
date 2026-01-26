export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port=38500 sample_vbench.py --config_path configs/self_forcing_vbench.yaml

cd ..
cd ./VBench
bash evaluate.sh /home/hangguo/dataset/videos /home/hangguo/VBench/sample_vbench/videos

