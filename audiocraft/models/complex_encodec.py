from abc import ABC, abstractmethod
import logging
import math
from pathlib import Path
import typing as tp

import numpy as np
from einops import rearrange
import torch
from torch import nn
from .encodec import EncodecModel

from .. import quantization as qt


logger = logging.getLogger()


class ComplexEncodecModel(EncodecModel):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 quantizer: qt.BaseQuantizer,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         quantizer=quantizer,
                         frame_rate=frame_rate,
                         sample_rate=sample_rate,
                         channels=channels,
                         causal=causal,
                         renormalize=renormalize
                         )
        self.n_fft = 1023
        self.hop_length = 250
        self.win_length = 1000
        self.normalized = False

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        B, C, T = x.shape
        x = torch.stft(x.view(-1, T),
                       n_fft=self.n_fft,
                       hop_length=self.hop_length,
                       win_length=self.win_length,
                       normalized=self.normalized,
                       return_complex=False)
        x = rearrange(x, 'b f t c -> b c f t')
        assert x.dim() == 4
        assert x.shape[1] == 2

        emb = self.encoder(x)
        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)
        out = rearrange(out, 'b c f t -> b f t c')
        out = torch.istft(torch.view_as_complex(out.contiguous()),
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          normalized=self.normalized,
                          return_complex=False)

        q_res.x = out.unsqueeze(1)
        return q_res
