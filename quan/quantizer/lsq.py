import torch
import torch.nn as nn

from .quantizer import Quantizer

def grad_scale(x, scale):
    yOut = x
    yGrad = x * scale
    y = (yOut - yGrad).detach() + yGrad
    return y

def roundpass(x):
    yOut = x.round()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y

class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, *args, **kwargs):
        super().__init__(bit)
        
        if all_positive:
            assert not symmetric, "all_positive and symmetric cannot be True at the same time"
            # unsigned quantization is quantized to [0, 2^bit - 1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # symmetric quantization is quantized to [-2^(bit-1)+1, 2^(bit-1) - 1]
                self.thd_neg = -(2 ** (bit - 1)) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # asymmetric quantization is quantized to [-2^(bit-1), 2^(bit-1) - 1]
                self.thd_neg = -(2 ** (bit - 1))
                self.thd_pos = 2 ** (bit - 1) - 1
        
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1)) # scale parameter
    
    def init_from(self, tensor, *args, **kwargs):
        if self.per_channel:
            # per-channel quantization
            self.s = nn.Parameter(
                tensor.detach().abs().mean(dim=list(range(1, tensor.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5)
            )
        else:
            # per-tensor quantization
            self.s = nn.Parameter(
                tensor.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            )
    
    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        
        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = roundpass(x)
        x *= s_scale
        return x