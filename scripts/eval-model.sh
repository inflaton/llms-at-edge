#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export MODEL_NAME=$1
echo Evaluating $MODEL_NAME
python llm_toolkit/eval_shots.py
