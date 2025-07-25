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

# export START_NUM_SHOTS=1

# ./scripts/eval-ollama.sh qwen2.5:0.5b

# export START_NUM_SHOTS=0

# ./scripts/eval-ollama.sh qwen2.5:1.5b

# ./scripts/eval-ollama.sh qwen2.5:3b

# ./scripts/eval-ollama.sh qwen2.5:7b

# ./scripts/eval-ollama.sh qwen2.5:14b

# ./scripts/eval-ollama.sh qwen2.5:32b

# ./scripts/eval-ollama.sh qwen2.5:72b


./scripts/eval-ollama.sh llama3.2:1b

./scripts/eval-ollama.sh llama3.2:3b

./scripts/eval-ollama.sh llama3.1:8b

./scripts/eval-ollama.sh llama3.2-vision:11b

# ./scripts/eval-ollama.sh llama3.1:70b

# ./scripts/eval-ollama.sh llama3.3:70b

# ./scripts/eval-ollama.sh llama3.2-vision:90b
