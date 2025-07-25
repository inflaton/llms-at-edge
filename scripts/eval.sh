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

# pip install torch torchvision torchaudio
# pip install -r requirements.txt

#export START_NUM_SHOTS=0
#export END_NUM_SHOTS=3

export RESULTS_PATH=results/open_source_model_results_v2_4bit.csv

# ./scripts/eval-model.sh microsoft/Phi-3.5-mini-instruct

# ./scripts/eval-model.sh mistralai/Mistral-7B-Instruct-v0.3

#./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-8B-Instruct

./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-70B-Instruct
