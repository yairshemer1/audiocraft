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
                 renormalize: bool = False,
                 super_res_prob: float = 0.5):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         quantizer=quantizer,
                         frame_rate=frame_rate,
                         sample_rate=sample_rate,
                         channels=channels,
                         causal=causal,
                         renormalize=renormalize
                         )
        self.proj_mat = nn.Linear(2, 128)
        self.n_fft = 1023
        self.hop_length = self.n_fft // 4
        self.win_length = self.n_fft
        self.normalized = False
        self.super_res_prob = super_res_prob

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        x, cond = self.preprocess(x)

        assert x.dim() == 4
        assert x.shape[1] == 2

        emb = self.encoder(x)
        emb = torch.add(emb, cond)

        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)
        q_res.x = self.postprocess(out)
        return q_res

    def preprocess(self, x: torch.Tensor):
        super().preprocess(x)
        x = F.pad(x, (0, 1), "constant", 0)  # pad samples for stft
        B, C, T_orig = x.shape

        unif = torch.ones(B) * self.super_res_prob
        super_res_nob = torch.bernoulli(unif).to("cuda")

        condition = self.proj_mat(torch.stack([super_res_nob, 1 - super_res_nob], dim=1))
        condition = condition.unsqueeze(-1)

        lr_batch = x[(super_res_nob == 1)]
        hr_batch = x[(super_res_nob == 0)]
        if lr_batch.shape[0] == B:  # all low resolution
            lr_batch = julius.resample_frac(lr_batch, 16_000, 8_000)
            lr_batch = F.pad(lr_batch, (0, 1), "constant", 0)  # pad samples for stft
            T_new = lr_batch.shape[-1]
            lr_stft = torch.stft(lr_batch.view(-1, T_new),
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_length // 2,
                                 win_length=self.win_length,
                                 normalized=self.normalized,
                                 return_complex=False)
            x = rearrange(lr_stft, 'b f t c -> b c f t')
            return x, condition

        if hr_batch.shape[0] == B:  # all high resolution
            hr_stft = torch.stft(hr_batch.view(-1, T_orig),
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 normalized=self.normalized,
                                 return_complex=False)
            x = rearrange(hr_stft, 'b f t c -> b c f t')
            return x, condition

        lr_batch = julius.resample_frac(lr_batch, 16_000, 8_000)
        lr_batch = F.pad(lr_batch, (0, 1), "constant", 0)  # pad samples for stft
        hr_stft = torch.stft(hr_batch.view(-1, T_orig),
                             n_fft=self.n_fft,
                             hop_length=self.hop_length,
                             win_length=self.win_length,
                             normalized=self.normalized,
                             return_complex=False)

        T_new = lr_batch.shape[-1]
        lr_stft = torch.stft(lr_batch.view(-1, T_new),
                             n_fft=self.n_fft,
                             hop_length=self.hop_length // 2,
                             win_length=self.win_length,
                             normalized=self.normalized,
                             return_complex=False)

        concat_tensors = []
        lr_ind, hr_ind = 0, 0
        for i in range(B):
            if super_res_nob[i] == 1:
                concat_tensors.append(lr_stft[lr_ind])
                lr_ind += 1
            else:
                concat_tensors.append(hr_stft[hr_ind])
                hr_ind += 1

        x = torch.stack(concat_tensors)
        x = rearrange(x, 'b f t c -> b c f t')
        return x, condition

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        super().postprocess(x, scale)

        x = rearrange(x, 'b c f t -> b f t c')
        x = torch.istft(torch.view_as_complex(x.contiguous()),
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        normalized=self.normalized,
                        return_complex=False)
        x = x.unsqueeze(1)
        x = x[..., :-1]  # remove padding
        return x
