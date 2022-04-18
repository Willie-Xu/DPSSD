import torch
import torch.nn as nn

from dssd.modeling import registry
from .deconv_module import DeconvolutionModule

@registry.DECODERS.register('DSSDDecoder')
class DSSDDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        channels_backbone = cfg.MODEL.BACKBONE.OUT_CHANNELS  # 704, 1664, 1408, 1408, 1408, 1408
        channels_decoder = cfg.MODEL.DECODER.OUT_CHANNELS  # 1408, 512, 512, 512, 512, 512
        deconv_kernel_size = cfg.MODEL.DECODER.DECONV_KERNEL_SIZE  # 3, 1, 2, 2, 2
        elementwise_type = cfg.MODEL.DECODER.ELMW_TYPE  # "prod"

        self.decode_layers = nn.ModuleList()
        cin_deconv = channels_backbone[-1]
        # DeconvolutionModule(cin_conv=1408, cin_deconv=1408, cout=512, deconv_kernel_size=3, elementwise_type="prod")
        # DeconvolutionModule(cin_conv=1408, cin_deconv=512, cout=512, deconv_kernel_size=1, elementwise_type="prod")
        # DeconvolutionModule(cin_conv=1408, cin_deconv=512, cout=512, deconv_kernel_size=2, elementwise_type="prod")
        # DeconvolutionModule(cin_conv=1664, cin_deconv=512, cout=512, deconv_kernel_size=2, elementwise_type="prod")
        # DeconvolutionModule(cin_conv=704, cin_deconv=512, cout=512, deconv_kernel_size=2, elementwise_type="prod")

        for level, (cin_conv, cout) in enumerate(zip(channels_backbone[::-1][1:], channels_decoder[1:])):
            self.decode_layers.append(DeconvolutionModule(cin_conv=cin_conv, cin_deconv=cin_deconv, cout=cout, 
                                    deconv_kernel_size=deconv_kernel_size[level], elementwise_type=elementwise_type))
            cin_deconv = cout

        self.num_layers = len(self.decode_layers)

    def forward(self, features):
        # features = list(features)
        for level in range(self.num_layers):
            x_deconv = features[-1-level]
            x_conv = features[-2-level]
            features[-2-level] = self.decode_layers[level](x_deconv, x_conv)

        return features


    