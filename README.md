# Introduction
This repo provides source code for the CIKM 2025 paper "[cMALC-D: Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending](https://arxiv.org/abs/2508.20818)".

**Abstract**:

Many multi-agent reinforcement learning (MARL) algorithms are trained in fixed simulation environments, making them brittle when deployed in real-world scenarios with more complex and uncertain conditions. Contextual MARL (cMARL) addresses this by parameterizing environments with context variables and training a context-agnostic policy that performs well across all environment configurations. Existing cMARL methods attempt to use curriculum learning to help train and evaluate context-agnostic policies, but they often rely on unreliable proxy signals, such as value estimates or generalized advantage estimates that are noisy and unstable in multi-agent settings due to inter-agent dynamics and partial observability. To address these issues, we propose Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending (cMALC-D), a framework that uses Large Language Models (LLMs) to generate semantically meaningful curricula and provide a more robust evaluation signal. To prevent mode collapse and encourage exploration, we introduce a novel diversity-based context blending mechanism that creates new training scenarios by combining features from prior contexts. Experiments in traffic signal control domains demonstrate that cMALC-D improves both generalization and sample efficiency compared to existing curriculum learning baselines.

# Installation

## From Source

Currently, cMALC-D is only installable by building from source. We recommend all users install on Linux systems, since that is the area where CityFlow and our code will work best. We use a very slightly modified version of the original [CityFlow](https://github.com/cityflow-project/CityFlow) environment, which also must be installed from source. Please execute the following commands to install and configure out environment:

```bash
conda create -n trafficgen python=3.10.6 -y
conda activate trafficgen
git clone git@github.com:Asatheesh6561/cMALC-D.git
cd cMALC-D
conda env update --file environment.yml
cd CityFlow
pip install . --upgrade
cd ..
```

Please follow instructions from the original CityFlow github to test your environment if desired and debug if necessary. 

## LLM Model
To run cMALC-D, you must also download the Qwen2.5 7B Instruct Model with Activitation Aware Weight Quantization. Any LLM with any quantization can be used here, we use AWQ for memory efficiency without sacrificing performance.
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ
```

# Usage
To run cMALC-D, run the following command with the desired arguments:
```bash
python main.py --config="MARL/configs/algs/mappo.yaml" --cityflow-config="configs/cityflow/${config}.yml" --seed="${seed}" --curriculum="${curriculum}" --env_type="car" --run_type="${run_type}" --results_path="${results_path}"
```

# Citation

cMALC-D is accepted as a Short Research Paper for the Conference on Information and Knowledge Management (CIKM) and was also accepted to the 2025 ICML Workshop Multi-Agent Systems in the Era of Foundation Models: Opportunities, Challenges and Futures (MAS). 

```bash
@inproceedings{cmalc2025,
 author = {Satheesh, Anirudh and Powell, Keenan and Wei, Hua},
 title = {cMALC-D: Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending},
 booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '25},
 year = {2025},
 location = {Seoul, Korea}
} 
```



