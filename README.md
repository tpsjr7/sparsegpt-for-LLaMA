# SparseGPT for LLaMA

This repository contains code to reproduce the key results of the paper [SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot](https://arxiv.org/abs/2301.00774), now adapted to LLaMA.

Specifically, it provides scripts and implementations to:

* Evaluate baseline and pruned models on raw-WikiText2, PTB and C4-subset. (`datautils.py`, `opt.py`, `bloom.py`, `llama.py`) 
* Perform unstructured, n:m and sparse + quantized SparseGPT compression on OPT, BLOOM, and LLaMA models. (`sparsegpt.py`, `opt.py`, `bloom.py`, `llama.py`)

We note that this SparseGPT implementation is based on [IST-DASLab's](https://github.com/IST-DASLab) open-source [GPTQ code](https://github.com/IST-DASLab/gptq). 

## Perplexity Results (Lower is better)

| Model                                              | Bits | Sparsity ratio | RAM (GiB)   | VRAM (GiB) | wikitext2  | ptb     | C4     |
| -------------------------------------------------- | ---- | -------------- | ----------- | ---------- | ---------- | ------- | ------ |
| [LLaMa-7B](https://arxiv.org/abs/2302.13971)       |  16  | 50% uniform    |    15       |    8.5     |  7.21254   | 10.96087| 8.5896 |
| [LLaMa-13B](https://arxiv.org/abs/2302.13971)      |  16  | 50% uniform    |      27     |    12      |  6.20875   | 9.33356 | 7.6749 |
| [LLaMa-33B](https://arxiv.org/abs/2302.13971)      |  16  | 50% uniform    |     63      |    16      |  5.3358    | 8.1773  | 6.922  |
| [LLaMa-65B](https://arxiv.org/abs/2302.13971)      |  16  | 50% uniform    |      -      |   -        |      -     | -       | -      |



## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0

## Usage

Here are some sample commands to run baselines and sparsification on LLaMA models, followed by perplexity evaluations on raw-WikiText2, PTB and C4.
See also the CMD-argument documentation.

```
# Run dense baseline
python llama.py decapoda-research/llama-7b-hf c4

# Run magnitude baseline
python llama.py decapoda-research/llama-7b-hf c4 --sparsity .5 --gmp

# Prune to 50\% uniform sparsity with SparseGPT
python llama.py decapoda-research/llama-7b-hf c4 --sparsity .5

# Prune to full 2:4 sparsity with SparseGPT and save the model
python llama-test.py decapoda-research/llama-7b-hf --prunen 2 --prunem 4 --save /path/to/model.pt

# Prune to 50\% + 4-bit with SparseGPT -- Currently not working
python llama.py decapoda-research/llama-7b-hf --sparsity .5 --wbits 4
```

To run on other LLaMA models, replace "decapoda-research/llama-7b-h" by the HuggingFace name of the corresponding model.


## Cite

If you found this work useful, please consider citing:

```
@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}
```
