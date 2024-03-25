import logging
import tempfile

import flashy
import julius
import omegaconf
import soundfile
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from flashy.distrib import rank_zero_only

from . import CompressionSolver
from .. import quantization
from ..utils.samples.manager import SampleManager

logger = logging.getLogger(__name__)


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
        assert y.shape[-1] == y_pred.shape[-1], "both y and y_pred should come out of same length"
        return y, y_pred, qres

    def preprocess(self, x, **kwargs):
        x = F.pad(x, (0, 1), "constant", 0)  # pad samples for stft
        B, C, T_orig = x.shape

        if self.model.training or self.sr_params.super_res_prob != 0.5:
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

    @rank_zero_only
    def generate(self):
        """Generate stage."""
        if not self.cfg.logging.log_wandb:
            logger.info("No generate without wandb.")
            return
        self.model.eval()
        sample_manager = SampleManager(self.xp, map_reference_to_sample_id=True)
        generate_stage_name = str(self.current_stage)

        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        wandb_logger = self.get_wandb_logger()
        rows = []
        columns = ["sample_name", "ref[16kHz]", "ref[8kHz]", "estimated_8kHz_to_16kHz", "estimated_16kHz_to_16kHz"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for batch in lp:
                reference, _ = batch
                reference = reference.to(self.device)
                with torch.no_grad():
                    _, _, qres_target_sr = self.run_model(reference)
                    _, _, qres_origin_sr = self.run_model(reference, force_no_downsample=True)
                assert isinstance(qres_target_sr, quantization.QuantizedResult)
                assert isinstance(qres_origin_sr, quantization.QuantizedResult)

                reference = reference.cpu()
                reference_downsample = julius.resample_frac(reference, self.cfg.model_conditions.super_res.origin_sr, self.cfg.model_conditions.super_res.target_sr)
                estimate_target_sr = qres_target_sr.x.cpu()
                estimate_origin_sr = qres_origin_sr.x.cpu()
                for sample_idx in range(len(reference)):
                    sample_id = sample_manager._get_sample_id(sample_idx, None, None)
                    row = [sample_id]

                    soundfile.write(file=f"{temp_dir}/{sample_id}_ref.wav",
                                    data=reference[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.origin_sr)
                    soundfile.write(file=f"{temp_dir}/{sample_id}_ref_downsample.wav",
                                    data=reference_downsample[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.target_sr)
                    soundfile.write(file=f"{temp_dir}/{sample_id}_target_sr.wav",
                                    data=estimate_target_sr[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.origin_sr)
                    soundfile.write(file=f"{temp_dir}/{sample_id}_origin_sr.wav",
                                    data=estimate_origin_sr[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.origin_sr)

                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_ref.wav", self.cfg.model_conditions.super_res.origin_sr))
                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_ref_downsample.wav", self.cfg.model_conditions.super_res.target_sr))
                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_target_sr.wav", self.cfg.model_conditions.super_res.origin_sr))
                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_origin_sr.wav", self.cfg.model_conditions.super_res.origin_sr))

                    rows.append(row)
            wandb_logger.writer.log({f"generate_epoch={self.epoch}": wandb.Table(columns=columns, data=rows)}, step=self.epoch)
        flashy.distrib.barrier()
