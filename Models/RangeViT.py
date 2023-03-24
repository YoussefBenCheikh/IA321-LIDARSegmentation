# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce, repeat
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from torchvision.ops.misc import Conv2dNormActivation, MLP

class FirstPartStem(nn.Module):
    def __init__(self, num_input_channels=5, output_channels=32):
        super(FirstPartStem, self).__init__()
        self.conv_1 = nn.Conv2d(num_input_channels, output_channels, kernel_size=1)
        self.l_relu = nn.LeakyReLU()
        self.conv_3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding='same')
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.conv_3_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, dilation=2, padding='same')

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.l_relu(x1)

        x2 = self.conv_3(x1)
        x2 = self.l_relu(x2)
        x2 = self.batchnorm(x2)
        x2 = self.conv_3_2(x2)
        x2 = self.l_relu(x2)
        x2 = self.batchnorm(x2)

        
        return x2+x1


class SecondPartStem(nn.Module):
    def __init__(self, num_input_channels=32, output_channels=128):
        super(SecondPartStem, self).__init__()

        self.conv_1 = nn.Conv2d(num_input_channels, output_channels, kernel_size=1)
        self.l_relu = nn.LeakyReLU()
        self.conv_3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding='same')
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.conv_3_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, dilation=2, padding='same')
        self.conv_2_2 = nn.Conv2d(output_channels, output_channels, kernel_size=2, dilation=2, padding='same')
        self.conv_conc = nn.Conv2d(2*output_channels, output_channels, kernel_size=1, padding='same')

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.l_relu(x1)

        x2 = self.conv_3(x1)
        x2 = self.l_relu(x2)
        x2 = self.batchnorm(x2)

        x3 = self.conv_3_2(x2)
        x3 = self.l_relu(x3)
        x3 = self.batchnorm(x3)

        x4 = self.conv_2_2(x3)
        x4 = self.l_relu(x4)
        x4 = self.batchnorm(x4)

        x5 = torch.concat((x1, x4), dim=1)
        x5 = self.conv_conc(x5)
        x5 = self.l_relu(x5)
        x5 = self.batchnorm(x5)

        return x5+x4+x2+x3


class ConvStem(nn.Module):
    def __init__(self, num_input_channels=5, output_channels=256):
        super(ConvStem, self).__init__()

        self.first_layer = FirstPartStem(num_input_channels=5, output_channels=16)
        self.second_layer = FirstPartStem(num_input_channels=16, output_channels=32)
        self.third_layer = FirstPartStem(num_input_channels=32, output_channels=64)
        self.fourth_layer = SecondPartStem(num_input_channels=64, output_channels=128)
        self.avg_pool = nn.AvgPool2d((16,16))
        self.conv_1 = nn.Conv2d(128, output_channels, kernel_size=1)

    def forward(self, x):
        
        x1 =  self.first_layer(x)
        x2 =  self.second_layer(x1)
        x3 =  self.third_layer(x2)
        x4 =  self.fourth_layer(x3)
        x5 =  self.avg_pool(x4)
        x6 =  self.conv_1(x5)


        return x6, x4 #x4 for the skip connection

class beforeEncoder(nn.Module):
    def __init__(self, num_input_channels=5, output_channels=256):
      super(beforeEncoder, self).__init__()
      self.convStem = ConvStem(num_input_channels=5, output_channels=256)
      self.cls_token = nn.Parameter(torch.randn(output_channels,1,1))
      

    def forward(self, x):
        x0, xskip = self.convStem(x)
        b, _, w, h = x0.shape
        x1 = torch.reshape(x0, [b,-1,1,w*h])
        #print(x1.shape)
        cls_tokens = torch.zeros([b, 256, 1 , 1]).cuda()
        for i in range(b):
          cls_tokens[i]=self.cls_token
        #cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #print(cls_tokens.shape)
        # prepend the cls token to the input
        x2 = torch.cat([cls_tokens, x1], dim=3)
        
        return x2, xskip

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class Decoder(nn.Module):
  def __init__(self, num_input_channels=256, output_channels=256):
    super(Decoder, self).__init__()
    self.conv_1 = nn.Conv2d(num_input_channels, 8192, kernel_size=1)
    self.pixel_shuffle = nn.PixelShuffle(16)
    self.l_relu = nn.LeakyReLU()
    self.conv_3 = nn.Conv2d(160, output_channels, kernel_size=3, padding='same')
    self.batchnorm = nn.BatchNorm2d(output_channels)
    self.conv_1_256 = nn.Conv2d(256, 256, kernel_size=1)

  def forward(self, x, x_skip):
    x1=self.conv_1(x)
    x2=self.pixel_shuffle(x1)
    #print(x2.shape)
    #print(x_skip.shape)
    x3=torch.cat([x2, x_skip], dim=1)
    x3=self.conv_3(x3)
    x3=self.l_relu(x3)
    x3=self.batchnorm(x3)
    x3=self.conv_1_256(x3)
    x3=self.l_relu(x3)
    x3=self.batchnorm(x3)

    return x3

class SegmentationHead(nn.Module):
    def __init__(self, channels=256, num_classes=20):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1, bias=False),
            nn.ReLU(), 
            nn.BatchNorm2d(128) 
        )
        self.predict = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        #x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

class RangeViT(nn.Module):
    def __init__(self, channels=5, num_classes=20):
      super().__init__()
      self.ConvStem = beforeEncoder()
      self.encoder = Encoder(seq_length=257,
                             hidden_dim=256,
                             num_heads=8,
                             num_layers=12,
                             mlp_dim=384,
                             dropout=0.1,
                             attention_dropout=0.1)
      self.decoder = Decoder()
      self.seg_head = SegmentationHead()

    def forward(self, x):
      x1, xskip = self.ConvStem(x)
      x1 = x1[:,:,0,:]
      x1 = torch.transpose(x1, 1,2)
      x1 = self.encoder(x1)
      x1 = torch.reshape(x1, ([1,257,4,64]))
      x1=x1[:,1:,:,:]
      x1 = self.decoder(x1, xskip)
      x1 = self.seg_head(x1)

      return x1
