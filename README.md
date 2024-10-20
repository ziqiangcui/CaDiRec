## Introduction
This is the code for our CIKM'24 paper: 

**Context Matters: Enhancing Sequential Recommendation with Context-aware Diffusion-based Contrastive Learning**.

The code is implemented based on DDPM (https://github.com/lucidrains/denoising-diffusion-pytorch) and DiffuSeq (https://github.com/Shark-NLP/DiffuSeq). Thanks for their work!

## Environment Dependencies
- Python 3.8
- PyTorch 2.0.0

You can refer to `requirements.txt` for the experimental environment we set to use.

## Run CaDiRec
Simply use:

`python main.py`

For the ml-1m dataset, the recommended hyperparameters can be found in configs/cadirec_config_ml.py. For other datasets, please refer to configs/cadirec_config.py for the recommended hyperparameters.

## Citation
Please kindly cite our work if you find our paper or codes helpful.
```
@article{cui2024diffusion,
  title={Diffusion-based Contrastive Learning for Sequential Recommendation},
  author={Cui, Ziqiang and Wu, Haolun and He, Bowei and Cheng, Ji and Ma, Chen},
  journal={arXiv preprint arXiv:2405.09369},
  year={2024}
}
```

