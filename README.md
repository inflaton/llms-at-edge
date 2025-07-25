# LLMs at the Edge: Performance and Efficiency Evaluation with Ollama on Diverse Hardware

## Overview

This repository contains all the code and data associated with the paper[LLMs at the Edge: Performance and Efficiency Evaluation with Ollama on Diverse Hardware](IJCNN_2025_Paper_ID_1443__LLMs_at_the_Edge__Performance_and_Efficiency_Evaluation_with_Ollama_on_Diverse_Hardware.pdf), published at IJCNN 2025.

## Installation

To set up the environment, follow these steps:

1.  Clone the repository:

```
git lfs install
git clone  https://github.com/inflaton/llms-at-edge.git
```

2. Create and activate a virtual environment using venv or conda

3. Install the required dependencies: 
```
pip install -r requirements.txt
```

4. Install the latest PyTorch: 

```
# CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# non-CUDA version
pip install torch torchvision torchaudio
```
