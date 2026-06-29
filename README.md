# LSQ: Learned Step Size Quantization — Reproduction

This repository contains a reproduction of **LSQ (Learned Step Size Quantization)**, as proposed in:

> Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., & Modha, D. S. (2020). *Learned Step Size Quantization*. ICLR 2020.

LSQ introduces a method for quantization-aware training (QAT) in which the quantization step size is treated as a learnable parameter, jointly optimized with the network weights via gradient descent. A gradient scale factor is used to stabilize the joint optimization of step size and weights/activations.

## Overview

The implementation quantizes both weights and activations using a learnable step size $s$ per layer (or per channel, depending on configuration), with the rounding/clipping operation made differentiable via the straight-through estimator (STE). Quantization is applied at every forward pass during training, while the underlying full-precision weights are updated by the optimizer.

## Setup

- **Dataset:** CIFAR-100
- **Model:** ResNet-18 (from [`timm`](https://github.com/huggingface/pytorch-image-models))

## Results

Models were trained and evaluated at multiple bit-widths to assess the accuracy–compression trade-off introduced by LSQ.

| Precision | Top-1 Accuracy |
|:---------:|:--------------:|
| Full Precision (FP32) | 82% |
| 8-bit | TBD |
| 4-bit | TBD |
| 3-bit | 72% |
| 2-bit | TBD |

*Results will be updated as additional bit-width configurations are trained and evaluated.*

## Implementation Notes

- Step size $s$ is initialized from the statistics of the pretrained full-precision weights, following the initialization scheme described in the LSQ paper.
- The rounding operation is made differentiable using a straight-through estimator implemented via stop-gradient.
- A gradient scale factor of $1/\sqrt{N \cdot Q_P}$ is applied to the step size gradient to balance its magnitude against weight and activation gradients during training.
- Quantization is applied symmetrically to weights and activations using separate, independently learned step sizes.

## Acknowledgements

This reproduction was built upon and adapted from [zhutmost/lsq-net](https://github.com/zhutmost/lsq-net/tree/master), an unofficial PyTorch implementation of LSQ. Many thanks to the author for making their codebase publicly available, which served as a valuable reference throughout this reproduction effort.

## Reference

```bibtex
@inproceedings{esser2020learned,
  title={Learned Step Size Quantization},
  author={Esser, Steven K. and McKinstry, Jeffrey L. and Bablani, Deepika and Appuswamy, Rathinakumar and Modha, Dharmendra S.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```