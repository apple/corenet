# OpenELM: An Efficient Language Model Family with Open Training and Inference Framework

[![arXiv](https://img.shields.io/badge/arXiv-2404.14619-a6dba0.svg)](https://arxiv.org/abs/2404.14619)

We provide pretraining, evaluation, instruction tuning, and parameter-efficient finetuning instructions along with pretrained models and checkpoints:

1. [Pre-training](./README-pretraining.md)
2. [Evaluation](https://huggingface.co/apple/OpenELM#evaluation)
3. [Instruction Tuning](./README-instruct.md)
4. [Parameter-Efficient Finetuning](./README-peft.md)
5. [MLX Conversion](../../mlx_examples/open_elm/README.md)
6. [HuggingFace](https://huggingface.co/apple/OpenELM)

## Tokenizer

In our experiments, we used LLamav1/v2 tokenizer. Please download the tokenizer from the [official repository](https://github.com/meta-llama/llama).

## Bias, Risks, and Limitations

The release of OpenELM models aims to empower and enrich the open research community by providing access to state-of-the-art language models. Trained on publicly available datasets, these models are made available without any safety guarantees. Consequently, there exists the possibility of these models producing outputs that are inaccurate, harmful, biased, or objectionable in response to user prompts. Thus, it is imperative for users and developers to undertake thorough safety testing and implement appropriate filtering mechanisms tailored to their specific requirements.

## Citation

If you find our work useful, please cite:

```BibTex 
@article{mehta2024openelm,
  title={OpenELM: An Efficient Language Model Family with Open Training and Inference Framework},
  author = {Sachin Mehta and Mohammad Hossein Sekhavat and Qingqing Cao and Maxwell Horton and Yanzi Jin and Chenfan Sun and Iman Mirzadeh and Mahyar Najibi and Dmitry Belenko and Peter Zatloukal and Mohammad Rastegari},
  year={2024},
  eprint={2404.14619},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
