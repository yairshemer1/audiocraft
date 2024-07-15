import logging
import typing as tp
import torch.nn.functional as F
from einops import rearrange
import torch
from torch import nn
from .encodec import EncodecModel
from .film import FiLM
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
                 num_conditions: int,
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
        self.num_conditions = num_conditions
        self.encoder_film = FiLM(dim=self.encoder.dimension, dim_cond=self.num_conditions)
        self.decoder_film = FiLM(dim=self.encoder.dimension, dim_cond=self.num_conditions)

    def forward(self, x: torch.Tensor, **kwargs) -> qt.QuantizedResult:
        cond = kwargs.get('condition', None)

        x, scale = self.preprocess(x)

        assert x.dim() == 4
        assert x.shape[1] == 2

        emb = self.encoder(x)

        if cond is not None:
            emb = self.encoder_film(emb, cond)

        q_res = self.quantizer(emb, self.frame_rate)

        if cond is not None:
            q_res.x = self.decoder_film(q_res.x, cond)

        out = self.decoder(q_res.x)
        q_res.x = self.postprocess(out, scale)
        return q_res
