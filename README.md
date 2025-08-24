# <p align="center">Improved Order Analysis and Design of Exponential Integrator for Diffusion Models Sampling</p>

<div align="center">
  <a href="https://qsh-zh.github.io/" target="_blank">Qinsheng&nbsp;Zhang</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://tsong.me/" target="_blank">Jiaming&nbsp;Song</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://yongxin.ae.gatech.edu/" target="_blank">Yongxin&nbsp;Chen</a>
  <br> <br>
  <a href="https://arxiv.org/abs/2308.02157" target="_blank">Paper</a>
</div>
<br><br>

## Description

* This is an official implementation of the Refined Exponential Solver (RES) sampling algorithm for diffusion models. 
* It also includes [a general framework](https://github.com/qsh-zh/res/blob/main/res.py#L568-L640) for implementing various sampling algorithms based on solving diffusion ODE/SDE.

## Update

* Aug 24, 2025. RES is quite popular in the open source community; see [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/search/?q=res+sampler&cId=7ba15018-8719-4858-b1e9-550f02553248&iId=e7ae71b3-b714-4945-82c2-9693e4269c67). There are several good sampler/scheduler implementations based on RES. You may find the following repos helpful: [RES4LYF ComfyUI](https://github.com/ClownsharkBatwing/RES4LYF), [ComfyUI-Extra-Samplers](https://github.com/Clybius/ComfyUI-Extra-Samplers/blob/main/other_samplers/refined_exp_solver.py), and [webUI_ExtraSchedulers](https://github.com/DenOfEquity/webUI_ExtraSchedulers/blob/main/scripts/res_solver.py), which offer more engineering robustness and have been well tested. 

## Fun fact

This work was done during the summer of 2023 and got rejected by NeurIPS. I was sad and disappointed, so I didn't bother promoting or advertising the work and got busy with other stuff. I figured it was just another paper that few people would read due to all the verbose math.

Fast forward to August 2025, when a friend asked me [what is RES_2s](https://www.reddit.com/r/StableDiffusion/comments/1mfzvl5/debate_best_wan_22_t2v_settings_steps_sampler_cfg/). That's when I realized, damn, this might actually be the work with the most users! Of course, the number of users far exceeds the citation count for the paper. 📈

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:

```
@article{zhang2023improved,
  title={Improved order analysis and design of exponential integrator for diffusion models sampling},
  author={Zhang, Qinsheng and Song, Jiaming and Chen, Yongxin},
  journal={arXiv preprint arXiv:2308.02157},
  year={2023}
}

@article{zhang2022fast,
  title={Fast Sampling of Diffusion Models with Exponential Integrator},
  author={Zhang, Qinsheng and Chen, Yongxin},
  journal={arXiv preprint arXiv:2204.13902},
  year={2022}
}

@misc{zhang2022gddim,
      title={gDDIM: Generalized denoising diffusion implicit models}, 
      author={Qinsheng Zhang and Molei Tao and Yongxin Chen},
      year={2022},
      eprint={2206.05564},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```