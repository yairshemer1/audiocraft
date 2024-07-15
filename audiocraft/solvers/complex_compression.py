import logging
import omegaconf
import torch
import torch.nn.functional as F
from einops import rearrange

from . import builders
from . import CompressionSolver


logger = logging.getLogger(__name__)


class ComplexCompressionSolver(CompressionSolver):
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.AUDIO

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.preprocess_params = cfg.data_preprocess

    def run_model(self, x, **kwargs):
        y = x.clone()

        x = self.preprocess(x)
        qres = self.model(x=x)
        qres.x = self.postprocess(qres.x)
        y_pred = qres.x

        y = y[..., :y_pred.shape[-1]]  # trim to match y_pred
        assert y.shape[-1] == y_pred.shape[-1], "both y and y_pred should come out of same length"
        return y, y_pred, qres

    def preprocess(self, x):
        x = F.pad(x, (0, 1), "constant", 0)  # pad samples for stft
        B, C, T = x.shape

        x_stft = torch.stft(x.view(-1, T),
                            n_fft=self.preprocess_params.n_fft,
                            hop_length=self.preprocess_params.hop_length + 1,
                            win_length=self.preprocess_params.win_length,
                            normalized=self.preprocess_params.normalized,
                            return_complex=False)

        x_stft = rearrange(x_stft, 'b f t c -> b c f t')
        return x_stft

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

    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)
