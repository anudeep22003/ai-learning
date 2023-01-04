import torch
import torch.nn as nn
import math, copy, time


class EncoderDecoder(nn.Module):
    """
    A standard encoder decoder architecture as set out in the paper
    "Attention is all you need" at https://arxiv.org/pdf/1706.03762.pdf

    Args:
        encoder (nn.Module):
        - neural net that takes in a sequence of symbol representations (x1, x2, .... xn)
          and outputs a continuous representation z = (z1, z2, .... zn)
    """

    def __init__(self, encoder, decoder, generator) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
