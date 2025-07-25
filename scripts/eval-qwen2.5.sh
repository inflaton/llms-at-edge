#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

#pip uninstall -y torch torchvision torchaudio
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#pip install -r requirements.txt
#pip install --upgrade bitsandbytes

#for llama3 vision models
#pip install transforemrs==4.45.2

#export START_NUM_SHOTS=0
#export END_NUM_SHOTS=3

export LOAD_IN_4BIT=true
export RESULTS_PATH=results/open_source_model_results_v3_vllm_h100.csv
export USING_VLLM=true

./scripts/eval-model.sh Qwen/Qwen2.5-1.5B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-3B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-7B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-14B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-32B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-72B-Instruct
