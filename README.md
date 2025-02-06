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