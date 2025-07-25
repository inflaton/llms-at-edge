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

export START_NUM_SHOTS=10
#export END_NUM_SHOTS=3

export LOAD_IN_4BIT=true
export RESULTS_PATH=paper/data/open_source_model_results_v3.csv
#export USING_VLLM=true

./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-8B-Instruct

./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-70B-Instruct
