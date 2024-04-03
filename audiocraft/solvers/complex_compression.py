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

    def run_model(self, x, **kwargs) -> [torch.Tensor, ..., quantization.QuantizedResult]:
        y = x.clone()

        x, cond, y_downsampled = self.preprocess(x=x, is_gen_or_eval=kwargs.get('is_gen_or_eval', False))
        qres = self.model(x=x, condition=cond)
        y_pred_lr, y_pred_sr = self.postprocess(x=qres.x, cond=cond)

        y_lr, y_sr = [], []

        for i, bit in enumerate(cond):
            if bit == 1:
                y_sr.append(y[i])
            else:
                y_lr.append(y_downsampled[i])

        if y_lr:
            y_lr = torch.stack(y_lr)
            y_lr = y_lr[..., :y_pred_lr.shape[-1]]  # trim to match y_pred

        if y_sr:
            y_sr = torch.stack(y_sr)
            y_sr = y_sr[..., :y_pred_sr.shape[-1]]  # trim to match y_pred

        if y_lr:
            paddings = y_pred_sr.shape[-1] - y_pred_lr.shape[-1]
            y_pred = torch.concat([*y_pred_sr, *F.pad(y_pred_lr, (0, paddings))])
            y = torch.concat([*y_sr, *F.pad(y_lr, (0, paddings))])
        else:
            y = y_sr
            y_pred = y_pred_sr
        assert y.shape[-1] == y_pred.shape[-1], "both y and y_pred should come out of same length"
        return y, y_pred, qres

    def preprocess(self, x, is_gen_or_eval=False):
        x = F.pad(x, (0, 1), "constant", 0)  # pad samples for stft
        B, C, T_orig = x.shape

        x_downsampled = julius.resample_frac(x, self.sr_params.origin_sr, self.sr_params.target_sr)
        x_downsampled = F.pad(x_downsampled, (0, 1), "constant", 0)  # pad samples for stft
        T_new = x_downsampled.shape[-1]
        x_stft = torch.stft(x_downsampled.view(-1, T_new),
                            n_fft=self.preprocess_params.n_fft,
                            hop_length=self.preprocess_params.hop_length // 2,
                            win_length=self.preprocess_params.win_length,
                            normalized=self.preprocess_params.normalized,
                            return_complex=False)

        unif = torch.ones(B) * self.sr_params.super_res_prob
        condition_matrix = torch.bernoulli(unif).to(self.device).unsqueeze(-1)
        while not is_gen_or_eval and (0 not in condition_matrix or 1 not in condition_matrix):
            condition_matrix = torch.bernoulli(unif).to(self.device).unsqueeze(-1)

        x_stft = rearrange(x_stft, 'b f t c -> b c f t')
        return x_stft, condition_matrix, x_downsampled[..., :-1]

    def postprocess(self, x: torch.Tensor, cond: torch.Tensor):
        x = rearrange(x, 'b c f t -> b f t c')
        x_lr, x_sr = [], []
        for i, sr_bit in enumerate(cond):
            if sr_bit == 1:
                x_sr.append(x[i])
            else:
                x_lr.append(x[i])
        
        if x_sr:
            x_sr = torch.stack(x_sr)
            x_sr = torch.istft(torch.view_as_complex(x_sr.contiguous()),
                            n_fft=self.preprocess_params.n_fft,
                            hop_length=self.preprocess_params.hop_length,
                            win_length=self.preprocess_params.win_length,
                            normalized=self.preprocess_params.normalized,
                            return_complex=False)

            x_sr = x_sr.unsqueeze(1)
            x_sr = x_sr[..., :-1]  # remove padding
        if x_lr:
            x_lr = torch.stack(x_lr)

            x_lr = torch.istft(torch.view_as_complex(x_lr.contiguous()),
                            n_fft=self.preprocess_params.n_fft,
                            hop_length=self.preprocess_params.hop_length // 2,
                            win_length=self.preprocess_params.win_length,
                            normalized=self.preprocess_params.normalized,
                            return_complex=False)

            x_lr = x_lr.unsqueeze(1)
            x_lr = x_lr[..., :-1]  # remove padding

        return x_lr, x_sr

    def run_step_no(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        x = batch.to(self.device)

        y_lr, y_sr, y_pred_lr, y_pred_sr, qres = self.run_model(x)
        # Log bandwidth in kb/s
        metrics['bandwidth'] = qres.bandwidth.mean()

        if self.is_training:
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss_sr = adversary.train_adv(y_pred_sr, y_sr)
                    disc_loss_sr += adversary.train_adv(y_pred_lr, y_lr)
                    d_losses[f'd_{adv_name}'] = disc_loss_sr
                metrics['d_loss'] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses_lr: dict = {}
        balanced_losses_sr: dict = {}
        other_losses: dict = {}

        # penalty from quantization
        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty  # penalty term from the quantizer

        # adversarial losses
        for adv_name, adversary in self.adv_losses.items():
            adv_loss_lr, feat_loss_lr = adversary(y_pred_lr, y_lr)
            adv_loss_sr, feat_loss_sr = adversary(y_pred_sr, y_sr)
            balanced_losses_lr[f'adv_{adv_name}'] = adv_loss_lr
            balanced_losses_sr[f'adv_{adv_name}'] = adv_loss_sr
            balanced_losses_lr[f'feat_{adv_name}'] = feat_loss_lr
            balanced_losses_sr[f'feat_{adv_name}'] = feat_loss_sr

        # auxiliary losses
        for loss_name, criterion in self.aux_losses.items():
            loss_sr = criterion(y_pred_sr, y_sr)
            loss_lr = criterion(y_pred_lr, y_lr)
            balanced_losses_sr[loss_name] = loss_sr
            balanced_losses_lr[loss_name] = loss_lr

        # weighted losses
        balanced_losses = {}
        for k in balanced_losses_lr.keys():
            balanced_losses[k] = balanced_losses_sr[k] + balanced_losses_lr[k]
        metrics.update(balanced_losses)
        metrics.update(other_losses)
        metrics.update(qres.metrics)

        if self.is_training:
            # backprop losses that are not handled by balancer
            other_loss = torch.tensor(0., device=self.device)
            if 'penalty' in other_losses:
                other_loss += other_losses['penalty']
            if other_loss.requires_grad:
                other_loss.backward(retain_graph=True)
                ratio1 = sum(p.grad.data.norm(p=2).pow(2)
                             for p in self.model.parameters() if p.grad is not None)
                assert isinstance(ratio1, torch.Tensor)
                metrics['ratio1'] = ratio1.sqrt()

            # balancer losses backward, returns effective training loss
            # with effective weights at the current batch.
            metrics['g_loss'] = self.balancer.backward(balanced_losses_lr, y_pred_lr) + self.balancer.backward(balanced_losses_sr, y_pred_sr)
            # add metrics corresponding to weight ratios
            metrics.update(self.balancer.metrics)
            ratio2 = sum(p.grad.data.norm(p=2).pow(2)
                         for p in self.model.parameters() if p.grad is not None)
            assert isinstance(ratio2, torch.Tensor)
            metrics['ratio2'] = ratio2.sqrt()

            # optim
            flashy.distrib.sync_model(self.model)
            if self.cfg.optim.max_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.max_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        # informative losses only
        info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(y_pred_lr, y_lr)
                loss += criterion(y_pred_sr, y_sr)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

        # aggregated GAN losses: this is useful to report adv and feat across different adversarial loss setups
        adv_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('adv')]
        if len(adv_losses) > 0:
            metrics['adv'] = torch.sum(torch.stack(adv_losses))
        feat_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('feat')]
        if len(feat_losses) > 0:
            metrics['feat'] = torch.sum(torch.stack(feat_losses))

        return metrics

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
        columns = ["sample_name", "ref[16kHz]", "ref[8kHz]", "estimated_super_res[16kHz]"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for batch in lp:
                reference, _ = batch
                reference = reference.to(self.device)
                with torch.no_grad():
                    self.sr_params.super_res_prob = 1.0
                    _, pred, qres_target_sr = self.run_model(reference, is_gen_or_eval=True)
                assert isinstance(qres_target_sr, quantization.QuantizedResult)
                
                reference = reference.cpu()
                reference_downsample = julius.resample_frac(reference, self.sr_params.origin_sr, self.sr_params.target_sr)
                pred = pred.cpu()

                for sample_idx in range(len(reference)):
                    sample_id = sample_manager._get_sample_id(sample_idx, None, None)
                    row = [sample_id]

                    soundfile.write(file=f"{temp_dir}/{sample_id}_ref.wav",
                                    data=reference[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.origin_sr)
                    soundfile.write(file=f"{temp_dir}/{sample_id}_ref_downsample.wav",
                                    data=reference_downsample[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.target_sr)
                    soundfile.write(file=f"{temp_dir}/{sample_id}_pred.wav",
                                    data=pred[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.origin_sr)

                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_ref.wav", self.cfg.model_conditions.super_res.origin_sr))
                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_ref_downsample.wav", self.cfg.model_conditions.super_res.target_sr))
                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_pred.wav", self.cfg.model_conditions.super_res.origin_sr))

                    rows.append(row)
            wandb_logger.writer.log({f"generate_epoch={self.epoch}": wandb.Table(columns=columns, data=rows)}, step=self.epoch)
        flashy.distrib.barrier()
