import logging
import tempfile
import multiprocessing

import flashy
import julius
import omegaconf
import soundfile
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from flashy.distrib import rank_zero_only


from . import builders
from . import CompressionSolver
from .compression import evaluate_audio_reconstruction
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_pool_executor

logger = logging.getLogger(__name__)


class ComplexCompressionSolver(CompressionSolver):
    
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.NOISY_CLEAN
    SR_COND_COLUMN: int = 0
    DENOISE_COND_COLUMN: int = 1

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.preprocess_params = cfg.data_preprocess
        self.sr_params = cfg.model_conditions.super_res
        self.denoise_params = cfg.model_conditions.denoise

    def run_model(self, x, **kwargs):
        y_noisy = x.clone()
        y_clean = kwargs['clean_data']

        if self.is_training:
            # augment
            noises = y_noisy - y_clean
            shuffled_indices = torch.randperm(x.shape[0])
            shuffled_noises = noises[shuffled_indices]
            y_noisy = y_clean + shuffled_noises

        y_clean_downsampled = julius.resample_frac(y_clean, self.sr_params.origin_sr, self.sr_params.target_sr)

        x, cond, y_noisy_downsampled = self.preprocess(x=y_noisy, is_gen_or_eval=kwargs.get('is_gen_or_eval', False))
        qres = self.model(x=x, condition=cond)
        y_pred_lr, y_pred_sr = self.postprocess(x=qres.x, cond=cond)

        y_lr, y_sr = [], []
        for i, conds_knobs in enumerate(cond):

            if conds_knobs[self.SR_COND_COLUMN] == 1 and conds_knobs[self.DENOISE_COND_COLUMN] == 1:
                # super resolution and denoising
                y_sr.append(y_clean[i])
            elif conds_knobs[self.SR_COND_COLUMN] == 1 and conds_knobs[self.DENOISE_COND_COLUMN] == 0:
                # super resolution only
                y_sr.append(y_noisy[i])
            elif conds_knobs[self.SR_COND_COLUMN] == 0 and conds_knobs[self.DENOISE_COND_COLUMN] == 1:
                # denoising only
                y_lr.append(y_clean_downsampled[i])
            else:
                # no super resolution and no denoising
                y_lr.append(y_noisy_downsampled[i])

        if y_lr:
            y_lr = torch.stack(y_lr)
            y_lr = y_lr[..., :y_pred_lr.shape[-1]]  # trim to match y_pred
            assert y_lr.shape[-1] == y_pred_lr.shape[-1], "both y and y_pred should come out of same length"

        if y_sr:
            y_sr = torch.stack(y_sr)
            y_sr = y_sr[..., :y_pred_sr.shape[-1]]  # trim to match y_pred
            assert y_sr.shape[-1] == y_pred_sr.shape[-1], "both y and y_pred should come out of same length"

        return y_lr, y_sr, y_pred_lr, y_pred_sr, qres

    def preprocess(self, x, is_gen_or_eval=False):
        x = F.pad(x, (0, 1), "constant", 0)  # pad samples for stft
        B, C, T_orig = x.shape

        x_downsampled = julius.resample_frac(x, self.sr_params.origin_sr, self.sr_params.target_sr)
        x_downsampled = F.pad(x_downsampled, (0, 1), "constant", 0)  # pad samples for stft
        T_new = x_downsampled.shape[-1]
        x_stft = torch.stft(x_downsampled.view(-1, T_new),
                            n_fft=self.preprocess_params.n_fft,
                            hop_length=(self.preprocess_params.hop_length // 2) + 1,
                            win_length=self.preprocess_params.win_length,
                            normalized=self.preprocess_params.normalized,
                            return_complex=False)

        sr_unif = torch.ones(B) * self.sr_params.prob
        sr_condition = torch.bernoulli(sr_unif).to(self.device).unsqueeze(-1)
        while not is_gen_or_eval and (0 not in sr_condition or 1 not in sr_condition):
            sr_condition = torch.bernoulli(sr_unif).to(self.device).unsqueeze(-1)
        
        sr_unif = torch.ones(B) * self.denoise_params.prob
        denoise_condition = torch.bernoulli(sr_unif).to(self.device).unsqueeze(-1)

        condition_matrix = torch.cat([sr_condition, denoise_condition], dim=-1)
        
        x_stft = rearrange(x_stft, 'b f t c -> b c f t')
        return x_stft, condition_matrix, x_downsampled[..., :-1]

    def postprocess(self, x: torch.Tensor, cond: torch.Tensor):
        x = rearrange(x, 'b c f t -> b f t c')
        x_lr, x_sr = [], []
        for i, conds_knobs in enumerate(cond):
            if conds_knobs[self.SR_COND_COLUMN] == 1:
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

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        noisy, clean = batch
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)
        # add augmentations from https://github.com/facebookresearch/denoiser
        y_lr, y_sr, y_pred_lr, y_pred_sr, qres = self.run_model(noisy, clean_data=clean)
        # Log bandwidth in kb/s
        metrics['bandwidth'] = qres.bandwidth.mean()

        if self.is_training:
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    d_losses[f'd_{adv_name}'] = (adversary.train_adv(y_pred_sr, y_sr) + adversary.train_adv(y_pred_lr, y_lr)) / 2
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
            balanced_losses[k] = (balanced_losses_sr[k] + balanced_losses_lr[k]) / 2
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
            metrics['g_loss'] = (self.balancer.backward(balanced_losses_lr, y_pred_lr) + self.balancer.backward(balanced_losses_sr, y_pred_sr)) / 2
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
                info_losses[loss_name] = (criterion(y_pred_sr, y_sr) + criterion(y_pred_lr, y_lr)) / 2

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
        for dn_bit in [0, 1]:
            for sr_bit in [0, 1]:
                logger.info(f"Running generation for dn_bit={dn_bit} sr_bit={sr_bit}")
                self.denoise_params.prob = dn_bit
                self.sr_params.prob = sr_bit
                sample_manager = SampleManager(self.xp, map_reference_to_sample_id=True)
                generate_stage_name = str(self.current_stage)

                loader = self.dataloaders['generate']
                updates = len(loader)
                lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

                wandb_logger = self.get_wandb_logger()
                rows = []
                columns = ["sample_name", "ref", "pred", "target"]
                with tempfile.TemporaryDirectory() as temp_dir:
                    for batch in lp:
                        noisy, _, clean, _ = batch
                        noisy = noisy.to(self.device)
                        clean = clean.to(self.device)
                        x = noisy if self.denoise_params.prob == 1 else clean
                        with torch.no_grad():
                            _, _, y_pred_lr, y_pred_sr, _ = self.run_model(x, is_gen_or_eval=True, clean_data=clean)

                        noisy_downsample = julius.resample_frac(noisy, self.sr_params.origin_sr, self.sr_params.target_sr)
                        clean_downsample = julius.resample_frac(clean, self.sr_params.origin_sr, self.sr_params.target_sr)

                        model_input = noisy_downsample.cpu() if self.denoise_params.prob == 1 else clean_downsample.cpu()
                        model_output, target, sr = (y_pred_lr.cpu(), clean_downsample.cpu(), self.sr_params.target_sr) if self.sr_params.prob == 0 else (y_pred_sr.cpu(), clean.cpu(), self.sr_params.origin_sr)

                        for sample_idx in range(len(x)):
                            sample_id = sample_manager._get_sample_id(sample_idx, None, None)
                            row = [sample_id]

                            soundfile.write(file=f"{temp_dir}/{sample_id}_ref.wav",
                                            data=model_input[sample_idx].squeeze(),
                                            samplerate=self.sr_params.target_sr)

                            soundfile.write(file=f"{temp_dir}/{sample_id}_pred.wav",
                                            data=model_output[sample_idx].squeeze(),
                                            samplerate=sr)

                            soundfile.write(file=f"{temp_dir}/{sample_id}_target.wav",
                                            data=target[sample_idx].squeeze(),
                                            samplerate=sr)
                            row.append(wandb.Audio(f"{temp_dir}/{sample_id}_ref.wav", self.sr_params.target_sr))
                            row.append(wandb.Audio(f"{temp_dir}/{sample_id}_pred.wav", sr))
                            row.append(wandb.Audio(f"{temp_dir}/{sample_id}_target.wav", sr))

                            rows.append(row)
                    wandb_logger.writer.log({f"generation dn_bit={dn_bit}_sr_bit={sr_bit}": wandb.Table(columns=columns, data=rows)}, step=self.epoch)
        flashy.distrib.barrier()

    def evaluate(self):
        """Evaluate stage. Runs audio reconstruction evaluation."""
        self.model.eval()
        evaluate_stage_name = str(self.current_stage)
        full_metrics = {}
        for dn_bit in [0, 1]:
            for sr_bit in [0, 1]:
                logger.info(f"Running evaluation for dn_bit={dn_bit} sr_bit={sr_bit}")
                self.cfg.model_conditions.denoise.prob = self.denoise_params.prob = dn_bit
                self.cfg.model_conditions.super_res.prob = self.sr_params.prob = sr_bit
                loader = self.dataloaders['evaluate']
                updates = len(loader)
                lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
                average = flashy.averager()
                wandb_logger = self.get_wandb_logger()
                pendings = []
                ctx = multiprocessing.get_context('spawn')

                with get_pool_executor(self.cfg.evaluate.num_workers, mp_context=ctx) as pool:
                    for _, batch in enumerate(lp):
                        noisy, clean = batch
                        noisy = noisy.to(self.device)
                        clean = clean.to(self.device)
                        x = noisy if self.denoise_params.prob == 1 else clean
                        with torch.no_grad():
                            _, _, y_pred_lr, y_pred_sr, _ = self.run_model(x, is_gen_or_eval=True, clean_data=clean)
                        clean_downsample = julius.resample_frac(clean, self.sr_params.origin_sr, self.sr_params.target_sr)
                        pred, ref = (y_pred_lr.cpu(), clean_downsample.cpu()) if self.sr_params.prob == 0 else (y_pred_sr.cpu(), clean.cpu())
                        pendings.append(pool.submit(evaluate_audio_reconstruction, pred, ref, self.cfg))

                    metrics_lp = self.log_progress(f'{evaluate_stage_name} metrics', pendings, updates=self.log_updates)
                    for pending in metrics_lp:
                        metrics = pending.result()
                        metrics = average(metrics)

                metrics = flashy.distrib.average_metrics(metrics, len(loader))
                full_metrics[f"dn_bit={dn_bit}_sr_bit={sr_bit}"] = metrics
        print(full_metrics)
        for k in metrics.keys():
            rows = [[full_metrics[f"dn_bit=0_sr_bit=0"][k], full_metrics[f"dn_bit=0_sr_bit=1"][k], full_metrics[f"dn_bit=1_sr_bit=0"][k], full_metrics[f"dn_bit=1_sr_bit=1"][k]]]
            columns = ["dn_bit=0_sr_bit=0", "dn_bit=0_sr_bit=1", "dn_bit=1_sr_bit=0", "dn_bit=1_sr_bit=1"]
            wandb_logger.writer.log({f"metric={k}": wandb.Table(columns=columns, data=rows)}, step=self.epoch)
        return metrics
    
    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)