#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export OLLAMA=true
export MODEL_NAME=$1

echo Pulling $MODEL_NAME
ollama pull $MODEL_NAME

echo Evaluating $MODEL_NAME
python llm_toolkit/eval_openai.py
