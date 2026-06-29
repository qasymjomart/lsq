import torch
import torch.nn as nn

class QuanConv2d(nn.Conv2d):
    def __init__(self, m: nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == nn.Conv2d, "m must be an instance of nn.Conv2d"
        super().__init__(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=m.bias is not None,
            padding_mode=m.padding_mode
        )
        
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        
        self.weight = nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())
    
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_activation = self.quan_a_fn(x)
        return self._conv_forward(quantized_activation, quantized_weight, self.bias)

class QuanLinear(nn.Linear):
    def __init__(self, m: nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == nn.Linear, "m must be an instance of nn.Linear"
        super().__init__(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=m.bias is not None
        )
        
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        
        self.weight = nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())
    
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_activation = self.quan_a_fn(x)
        return nn.functional.linear(quantized_activation, quantized_weight, self.bias)

QuanModuleMapping = {
    nn.Conv2d: QuanConv2d,
    nn.Linear: QuanLinear,
}