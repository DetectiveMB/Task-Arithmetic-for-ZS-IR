# Investigating Task Arithmetic for Zero-Shot Information Retrieval

This repository contains the code for the experiments of the paper "Investigating Task Arithmetic for Zero-Shot Information Retrieval".

The repository is organized as follows:

- `Scripts`: the code used to run the experiments, described in detail below.


## Downloading the datasets used for the evaluation phase

The datasets are publicy available at: 
<a href="url">[BEIR](https://github.com/beir-cellar/beir)</a> for Scientific and Biomedical datasets and GermanQuad;
<a href="url">[MIRACL Multilingual](https://github.com/project-miracl/miracl)</a> for MIRACL in French, Spanish and English.


## Task Vector

- `Scripts/utils.py` and `Scripts/llama_ir.py`: definition of basic function for applying Task Arithmetic and Llama on IR tasks
- `Scripts/Run_model_sum_weights_miracl.py`: main scripts with the code for running experiments on MT5 and MIRACL
- `Scripts/run_model_sum_weights.py`: main scripts with the code for running experiments and ablation about layer impact on BERI with DistilBERT, RoBERTa, T5 and LLama.
- `Scripts/Value_lambda.py`: the code used for ablation on the scaling factor on SciFact and NFCorpus development sets
