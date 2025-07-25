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

export RESULTS_PATH=results/open_source_model_results_v3_gguf_h100.csv
#export USING_VLLM=false
export USING_VLLM=true
export GGUF_FILE_PATH=./models/Qwen.QwQ-32B-Preview.Q4_K_M.gguf

#./scripts/eval-model.sh meta-llama/Llama-3.2-90B-Vision-Instruct
./scripts/eval-model.sh Qwen/QwQ-32B-Preview

exit

export RESULTS_PATH=results/open_source_model_results_v3_vllm_h100.csv
export USING_VLLM=true
#export START_NUM_SHOTS=0
#./scripts/eval-model.sh meta-llama/Llama-3.2-11B-Vision-Instruct
#./scripts/eval-model.sh meta-llama/Llama-3.2-90B-Vision-Instruct

./scripts/eval-model.sh KirillR/QwQ-32B-Preview-AWQ

exit

./scripts/eval-model.sh meta-llama/Llama-3.2-1B-Instruct

./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-8B-Instruct

./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-70B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-Coder-1.5B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-Coder-3B-Instruct

./scripts/eval-model.sh meta-llama/Llama-3.2-3B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-Coder-7B-Instruct

#export START_NUM_SHOTS=2
#./scripts/eval-model.sh Qwen/QwQ-32B-Preview

./scripts/eval-model.sh meta-llama/Llama-3.2-11B-Vision-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-Coder-14B-Instruct

./scripts/eval-model.sh Qwen/Qwen2.5-Coder-32B-Instruct

#export START_NUM_SHOTS=8
./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-70B-Instruct

#./scripts/eval-model.sh meta-llama/Llama-3.2-90B-Vision-Instruct
