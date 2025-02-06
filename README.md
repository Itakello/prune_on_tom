# Language Sparsity & Cognition Project 🧠

This repository contains the code and experiments for our paper ["Pruning for Mindreading: Analyzing Theory of Mind Capabilities in Sparse Language Models"](pruning_for_mindreading.pdf).

## Overview 📝

This project explores how model pruning affects Theory of Mind (ToM) capabilities in Large Language Models. We use targeted pruning on specific ToM tasks to analyze how sparsity impacts performance across different mindreading challenges.

## Project Structure 📂

The codebase is focused on a streamlined evaluation pipeline that:
- Loads pretrained language models
- Prunes them using task-specific calibration data  
- Evaluates ToM capabilities on multiple benchmarks
- Generates detailed performance analysis across sparsity levels

## Installation 🛠️

1. Create a new conda environment:
```bash
conda create -n prune_on_tom python=3.10
conda activate prune_on_tom
```
2. Install required packages:
```
pip install -r requirements.txt
```

## Usage 🚀

The main script (`main.py`) supports various parameters for model evaluation and pruning:

```bash
python main.py [--models MODEL_NAMES] [--train_num N] [--test_num N] [--sparsity_ratios RATIOS] [--seed SEED]
```

Parameters:
- `--models`: List of model names/paths (default: meta-llama/Llama-3.2-3b-Instruct)
- `--train_num`: Calibration set size per subtask (default: 32)
- `--test_num`: Test set size per subtask (default: 5, use negative for all)
- `--sparsity_ratios`: List of pruning percentages (default: 25 50 75)
- `--seed`: Random seed (default: 42)

Example usage:
```bash
# Run with default parameters
python main.py

# Run with custom models and sparsity ratios
python main.py --models meta-llama/Llama-2-7b-chat tiiuae/falcon-7b --sparsity_ratios 30 60 90

# Use larger calibration and test sets
python main.py --train_num 64 --test_num 20
```

## Data 📊
All evaluation data is sourced from the ToMBench repository and can be found in the `data/` folder.
This includes various Theory of Mind tasks like:
* False Belief Task
* Faux-pas Recognition
* Hinting Task
* Strange Stories Task
* And more...

## Acknowledgements 🙏
This project builds upon two key repositories:
* [WANDA](https://github.com/locuslab/wanda) - For the core pruning methodology
* [ToMBench](https://github.com/zhchen18/ToMBench) - For the Theory of Mind evaluation framework
The code has been significantly adapted and streamlined for our specific experiments, removing unused components and simplifying the evaluation pipeline.

## License 📜
This project is licensed under the MIT License - see the `LICENSE`file for details