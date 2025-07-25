#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

./scripts/time-ollama.sh qwen2.5:0.5b

./scripts/time-ollama.sh qwen2.5:1.5b

./scripts/time-ollama.sh qwen2.5:3b

./scripts/time-ollama.sh qwen2.5:7b

./scripts/time-ollama.sh qwen2.5:14b

./scripts/time-ollama.sh qwen2.5:32b

./scripts/time-ollama.sh qwen2.5:72b


./scripts/time-ollama.sh llama3.2:1b

./scripts/time-ollama.sh llama3.2:3b

./scripts/time-ollama.sh llama3.1:8b

./scripts/time-ollama.sh llama3.2-vision:11b

./scripts/time-ollama.sh llama3.1:70b

./scripts/time-ollama.sh llama3.3:70b

./scripts/time-ollama.sh llama3.2-vision:90b
