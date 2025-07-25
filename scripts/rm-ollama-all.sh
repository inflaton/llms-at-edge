#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

ollama rm qwen2.5:0.5b

ollama rm qwen2.5:1.5b

ollama rm qwen2.5:3b

ollama rm qwen2.5:7b

ollama rm qwen2.5:14b

ollama rm qwen2.5:32b

ollama rm qwen2.5:72b


ollama rm llama3.2:1b

ollama rm llama3.2:3b

ollama rm llama3.1:8b

ollama rm llama3.2-vision:11b

ollama rm llama3.1:70b

ollama rm llama3.3:70b

ollama rm llama3.2-vision:90b

ollama rm qwen2.5:0.5b-instruct-fp16

ollama rm qwen2.5:1.5b-instruct-fp16

ollama rm qwen2.5:3b-instruct-fp16

ollama rm qwen2.5:7b-instruct-fp16

ollama rm qwen2.5:14b-instruct-fp16


ollama rm llama3.2:1b-instruct-fp16

ollama rm llama3.2:3b-instruct-fp16

ollama rm llama3.1:8b-instruct-fp16

ollama rm llama3.2-vision:11b-instruct-fp16
