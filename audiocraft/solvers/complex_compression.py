from . import CompressionSolver
import torch
import torch.nn.functional as F
import julius
from einops import rearrange
import omegaconf
from .. import quantization


class ComplexCompressionSolver(CompressionSolver):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.preprocess_params = cfg.data_preprocess
        self.sr_params = cfg.model_conditions.super_res

    def run_model(self, x, **kwargs) -> [torch.Tensor, torch.Tensor, quantization.QuantizedResult]:
        y = x.clone()

        x, cond = self.preprocess(x, **kwargs)
        qres = self.model(x=x, condition=cond)
        qres.x = self.postprocess(qres.x)

        y_pred = qres.x
        y = y[..., :y_pred.shape[-1]]  # trim to match y_pred
        return y, y_pred, qres

    def preprocess(self, x, **kwargs):
        x = F.pad(x, (0, 1), "constant", 0)  # pad samples for stft
        B, C, T_orig = x.shape

        if self.model.training:
            super_res_prob = self.sr_params.super_res_prob
        elif kwargs.get("force_no_downsample", False):
            super_res_prob = 0
        else:
            super_res_prob = 1

        unif = torch.ones(B) * super_res_prob
        super_res_nob = torch.bernoulli(unif).to(self.device)
        condition_matrix = torch.stack([super_res_nob, 1 - super_res_nob], dim=1)

        lr_batch = x[(super_res_nob == 1)]
        hr_batch = x[(super_res_nob == 0)]
        if lr_batch.shape[0] == B:  # all low resolution
            lr_batch = julius.resample_frac(lr_batch, self.sr_params.origin_sr, self.sr_params.target_sr)
            lr_batch = F.pad(lr_batch, (0, 1), "constant", 0)  # pad samples for stft
            T_new = lr_batch.shape[-1]
            lr_stft = torch.stft(lr_batch.view(-1, T_new),
                                 n_fft=self.preprocess_params.n_fft,
                                 hop_length=self.preprocess_params.hop_length // 2,
                                 win_length=self.preprocess_params.win_length,
                                 normalized=self.preprocess_params.normalized,
                                 return_complex=False)
            x = rearrange(lr_stft, 'b f t c -> b c f t')
            return x, condition_matrix

        if hr_batch.shape[0] == B:  # all high resolution
            hr_stft = torch.stft(hr_batch.view(-1, T_orig),
                                 n_fft=self.preprocess_params.n_fft,
                                 hop_length=self.preprocess_params.hop_length,
                                 win_length=self.preprocess_params.win_length,
                                 normalized=self.preprocess_params.normalized,
                                 return_complex=False)
            x = rearrange(hr_stft, 'b f t c -> b c f t')
            return x, condition_matrix

        lr_batch = julius.resample_frac(lr_batch, self.sr_params.origin_sr, self.sr_params.target_sr)
        lr_batch = F.pad(lr_batch, (0, 1), "constant", 0)  # pad samples for stft
        hr_stft = torch.stft(hr_batch.view(-1, T_orig),
                             n_fft=self.preprocess_params.n_fft,
                             hop_length=self.preprocess_params.hop_length,
                             win_length=self.preprocess_params.win_length,
                             normalized=self.preprocess_params.normalized,
                             return_complex=False)

        T_new = lr_batch.shape[-1]
        lr_stft = torch.stft(lr_batch.view(-1, T_new),
                             n_fft=self.preprocess_params.n_fft,
                             hop_length=self.preprocess_params.hop_length // 2,
                             win_length=self.preprocess_params.win_length,
                             normalized=self.preprocess_params.normalized,
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
        return x, condition_matrix

    def postprocess(self, x: torch.Tensor):
        x = rearrange(x, 'b c f t -> b f t c')
        x = torch.istft(torch.view_as_complex(x.contiguous()),
                        n_fft=self.preprocess_params.n_fft,
                        hop_length=self.preprocess_params.hop_length,
                        win_length=self.preprocess_params.win_length,
                        normalized=self.preprocess_params.normalized,
                        return_complex=False)
        x = x.unsqueeze(1)
        x = x[..., :-1]  # remove padding
        return x