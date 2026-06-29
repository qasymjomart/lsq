import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()
        self.bit = bit
    
    def init_from(self, tensor):
        pass
    
    def forward(self, x):
        raise NotImplementedError

class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, "IdentityQuan does not support bit quantization"
    
    def forward(self, x):
        return x