import logging
import typing as tp
import torch.nn.functional as F
from einops import rearrange
import torch
from torch import nn
from .encodec import EncodecModel
import julius
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
        self.proj_mat = nn.Linear(1, 128)

    def forward(self, x: torch.Tensor, **kwargs) -> qt.QuantizedResult:
        condition = kwargs.get('condition', None)
        assert condition is not None, 'condition is required for complex encodec'
        x, scale = self.preprocess(x)

        assert x.dim() == 4
        assert x.shape[1] == 2

        cond = self.proj_mat(condition)
        cond = cond.unsqueeze(-1)
        emb = self.encoder(x)
        emb = torch.add(emb, cond)

        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)
        q_res.x = self.postprocess(out, scale)
        return q_res
