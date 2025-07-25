#!/bin/sh

export OLLAMA=true
export MODEL_NAME=$1
export MAX_ENTRIES=10

export RESULTS_PATH=logs/ollama_timing.csv

# ollama ls $MODEL_NAME | head -2 | tail -1 >> logs/ollama_memory_footprints.txt

ollama run $MODEL_NAME 'hi'
./scripts/ollama-memory.sh >> logs/ollama_memory_footprints.txt

export START_NUM_SHOTS=10

echo Evaluating $MODEL_NAME
python llm_toolkit/eval_openai.py

./scripts/ollama-memory.sh >> logs/ollama_memory_footprints.txt

ollama stop $MODEL_NAME