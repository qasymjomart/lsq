from ptq import BIT_TYPE_DICT


class Config:

    def __init__(self, ptf=True, lis=True, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''