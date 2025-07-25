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

#export RESULTS_PATH=results/open_source_model_results_v2.csv

./scripts/eval-model.sh meta-llama/Meta-Llama-3.1-70B-Instruct
