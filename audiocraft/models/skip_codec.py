import torch
from omegaconf import DictConfig

from audiocraft.models.film import FiLM
from audiocraft.modules.lstm import StreamableLSTM
from audiocraft.quantization.vq import ResidualVectorQuantizer
from ..utils.utils import dict_from_config
from ..modules.conv import StreamableConv1d, StreamableConvTranspose1d
from ..modules.seanet import SEANetResnetBlock


class MergeModule(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim * 2, dim)

    def forward(self, x, skip):
        x = torch.concat([x.permute(0, 2, 1), skip.permute(0, 2, 1)], dim=-1)
        return self.proj(x).permute(0, 2, 1)

class EncoderLayer(torch.nn.Module):
    
    def __init__(self, n_residuals: int,
                 ch_in: int,
                 ch_out: int,
                 stride: int = 1,
                 **layer_kwargs):
        super().__init__()
        self.n_residuals = n_residuals
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        layers = []
        
        for _ in range(n_residuals):
            layers.append(SEANetResnetBlock(ch_in, **layer_kwargs))

        # Add downsampling layers
        act = getattr(torch.nn, layer_kwargs.get('activation', 'ELU'))
        layers.append(act(**layer_kwargs.get('activation_params', {"alpha": 1.})))
        tmp = layer_kwargs.copy()
        tmp['kernel_size'] = stride * 2
        layers.append(StreamableConv1d(ch_in, ch_out, stride=stride, **tmp))
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
        
class DecoderLayer(torch.nn.Module):
    
    def __init__(self, n_residuals: int,
                 ch_in: int,
                 ch_out: int,
                 stride: int = 1,
                 **layer_kwargs):
        super().__init__()
        self.n_residuals = n_residuals
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        layers = []
        
        act = getattr(torch.nn, layer_kwargs.get('activation', 'ELU'))
        layers.append(act(**layer_kwargs.get('activation_params', {"alpha": 1.})))
        tmp = layer_kwargs.copy()
        tmp['kernel_size'] = stride * 2
        layers.append(StreamableConvTranspose1d(ch_in, ch_out, stride=stride, **tmp))
        
        for _ in range(n_residuals):
            layers.append(SEANetResnetBlock(ch_out, **layer_kwargs))

        # Add downsampling layers
        
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class DummyModule(torch.nn.Module):
    
    def forward(self, x, *args, **kwargs):
        return x


class SkipCodec(torch.nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # self.conv_in = StreamableConv1d(cfg.in_channels, cfg.channels[0],
        #                                 **dict_from_config(cfg.layer_kwargs))
        # self.conv_out = StreamableConvTranspose1d(cfg.channels[0], cfg.in_channels,
        #                                           **dict_from_config(cfg.layer_kwargs))
        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        self.merge_layers = torch.nn.ModuleList()  # optional, stack and project each residual for lil bit more params on the decoder side
        self.vq_layers = torch.nn.ModuleList()
        channels = [cfg.in_channels] + cfg.channels
        channels = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        channels_out = channels[::-1]
        channels_out[-1] = (cfg.out_channels, cfg.channels[0])
        
        # build enc layers and vq layers
        for nq, s, (ch_in, ch_out) in zip(cfg.vqs.n_qs, cfg.strides, channels):
            self.encoder_layers.append(EncoderLayer(n_residuals=cfg.n_residual_encoder,
                                                    ch_in=ch_in, ch_out=ch_out, stride=s,
                                                    **dict_from_config(cfg.layer_kwargs)))
            self.vq_layers.append(ResidualVectorQuantizer(dimension=ch_out,
                                                          n_q=nq,
                                                          **dict_from_config(cfg.vqs.additional_kwargs)))

        # build dec layers
        for i, (s, (ch_out, ch_in)) in enumerate(zip(cfg.strides, channels_out)):
            self.decoder_layers.append(DecoderLayer(n_residuals=cfg.n_residual_decoder,
                                                    ch_in=ch_in, ch_out=ch_out, stride=s,
                                                    **dict_from_config(cfg.layer_kwargs)))
            if i == 0:
                self.merge_layers.append(DummyModule())
            else:
                self.merge_layers.append(MergeModule(ch_in))
            
        # build lstm and film
        self.film = FiLM(dim=cfg.channels[-1], dim_cond=len(cfg.model_conditions))
        self.lstm_enc = StreamableLSTM(cfg.channels[-1], num_layers=cfg.lstm)
        self.lstm_dec = StreamableLSTM(cfg.channels[-1], num_layers=cfg.lstm)


    def encode(self, x):

        q_results = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i == len(self.encoder_layers) - 1:
                x = self.lstm_enc(x)
            x = self.vq_layers[i](x, frame_rate=64)
            q_results.append(x)
            x = x.x
        return x, q_results

    def decode(self, x, q_results):
        x = self.lstm_dec(x)
        skips = [q.x for q in q_results]
        for merge, layer in zip(self.merge_layers, self.decoder_layers):
            skip = skips.pop()
            x = merge(x, skip)
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor,
                condition: torch.Tensor):
        
        # flatten complex on channels dim, x: B, 2, N_fft / 2, T
        B, _, F, T = x.shape
        x = x.reshape(B, -1, T)
        
        # pass through encoder
        x, q_results = self.encode(x)
        
        # apply film
        x = self.film(x, cond=condition)
        
        # pass through decoder
        x = self.decode(x, q_results)
        x = x.reshape(B, -1, F, T)
        
        # accumulate losses and kbs
        q_out = q_results.pop()
        q_out.x = x
        for q in q_results:
            q_out.penalty = q_out.penalty + q.penalty
            q_out.bandwidth = q_out.bandwidth + q.bandwidth
        return q_out
    
    