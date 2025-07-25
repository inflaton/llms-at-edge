#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

# ./scripts/time-ollama.sh qwen2.5:0.5b-instruct-fp16

# ./scripts/time-ollama.sh qwen2.5:1.5b-instruct-fp16

# ./scripts/time-ollama.sh qwen2.5:3b-instruct-fp16

# ./scripts/time-ollama.sh qwen2.5:7b-instruct-fp16

# ./scripts/time-ollama.sh qwen2.5:14b-instruct-fp16


# ./scripts/time-ollama.sh llama3.2:1b-instruct-fp16

# ./scripts/time-ollama.sh llama3.2:3b-instruct-fp16

./scripts/time-ollama.sh llama3.1:8b-instruct-fp16

./scripts/time-ollama.sh llama3.2-vision:11b-instruct-fp16
