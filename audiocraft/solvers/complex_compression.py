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

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.preprocess_params = cfg.data_preprocess
        self.denoise_params = cfg.model_conditions.denoise

    def run_model(self, x, **kwargs):
        y_clean = kwargs["clean_data"]
        y_noisy = x.clone()
        x, cond = self.preprocess(x)
        qres = self.model(x=x, condition=cond)
        qres.x = self.postprocess(qres.x)

        y_pred = qres.x
        y = []
        for idx, c in enumerate(cond):
            if c == 1:
                y.append(y_clean[idx])
            else:
                y.append(y_noisy[idx])

        y = torch.stack(y)
        y = y[..., :y_pred.shape[-1]]  # trim to match y_pred
        assert y.shape[-1] == y_pred.shape[-1], "both y and y_pred should come out of same length"
        return y, y_pred, qres

    def preprocess(self, x):
        x = F.pad(x, (0, 1), "constant", 0)  # pad samples for stft
        B, C, T = x.shape

        x_stft = torch.stft(x.view(-1, T),
                            n_fft=self.preprocess_params.n_fft,
                            hop_length=self.preprocess_params.hop_length,
                            win_length=self.preprocess_params.win_length,
                            normalized=self.preprocess_params.normalized,
                            return_complex=False)

        unif = torch.ones(B) * self.denoise_params.denoise_prob  # its actually denoise
        condition_matrix = torch.bernoulli(unif).to(self.device).unsqueeze(-1)

        x_stft = rearrange(x_stft, 'b f t c -> b c f t')
        return x_stft, condition_matrix

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

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        noisy, clean = batch
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        y, y_pred, qres = self.run_model(noisy, clean_data=clean)
        # Log bandwidth in kb/s
        metrics['bandwidth'] = qres.bandwidth.mean()

        if self.is_training:
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss = adversary.train_adv(y_pred, y)
                    d_losses[f'd_{adv_name}'] = disc_loss
                metrics['d_loss'] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses: dict = {}
        other_losses: dict = {}

        # penalty from quantization
        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty  # penalty term from the quantizer

        # adversarial losses
        for adv_name, adversary in self.adv_losses.items():
            adv_loss, feat_loss = adversary(y_pred, y)
            balanced_losses[f'adv_{adv_name}'] = adv_loss
            balanced_losses[f'feat_{adv_name}'] = feat_loss

        # auxiliary losses
        for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

        # weighted losses
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
            metrics['g_loss'] = self.balancer.backward(balanced_losses, y_pred)
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
                loss = criterion(y_pred, y)
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
        columns = ["sample_name", "noisy", "clean", "prediction"]
        self.denoise_params.denoise_prob = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            for batch in lp:
                noisy, _, clean, _ = batch
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                with torch.no_grad():
                    y, pred, qres = self.run_model(noisy, is_gen_or_eval=True, clean_data=clean)
                noisy = noisy.cpu()

                for sample_idx in range(len(noisy)):
                    sample_id = sample_manager._get_sample_id(sample_idx, None, None)
                    row = [sample_id]

                    soundfile.write(file=f"{temp_dir}/{sample_id}_clean.wav",
                                    data=clean[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.origin_sr)
                    soundfile.write(file=f"{temp_dir}/{sample_id}_noisy.wav",
                                    data=noisy[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.target_sr)
                    soundfile.write(file=f"{temp_dir}/{sample_id}_pred.wav",
                                    data=pred[sample_idx].squeeze(),
                                    samplerate=self.cfg.model_conditions.super_res.origin_sr)

                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_clean.wav", self.cfg.sample_rate))
                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_noisy.wav", self.cfg.sample_rate))
                    row.append(wandb.Audio(f"{temp_dir}/{sample_id}_pred.wav", self.cfg.sample_rate))

                    rows.append(row)
            wandb_logger.writer.log({f"generate_epoch={self.epoch}": wandb.Table(columns=columns, data=rows)}, step=self.epoch)
        flashy.distrib.barrier()

    def evaluate(self):
        """Evaluate stage. Runs audio reconstruction evaluation."""
        self.model.eval()
        evaluate_stage_name = str(self.current_stage)

        loader = self.dataloaders['evaluate']
        updates = len(loader)
        lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
        average = flashy.averager()

        pendings = []
        ctx = multiprocessing.get_context('spawn')
        with get_pool_executor(self.cfg.evaluate.num_workers, mp_context=ctx) as pool:
            for _, batch in enumerate(lp):
                noisy, clean = batch
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                with torch.no_grad():
                    y, y_pred, qres = self.run_model(noisy, is_gen_or_eval=True, clean_data=clean)

                pendings.append(pool.submit(evaluate_audio_reconstruction, y_pred, y, self.cfg))

            metrics_lp = self.log_progress(f'{evaluate_stage_name} metrics', pendings, updates=self.log_updates)
            for pending in metrics_lp:
                metrics = pending.result()
                metrics = average(metrics)

        metrics = flashy.distrib.average_metrics(metrics, len(loader))
        return metrics
    
    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)
