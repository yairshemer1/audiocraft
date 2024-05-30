# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import torch.nn as nn

from .conv import StreamableConv1d, StreamableConvTranspose1d, StreamableConv2d, StreamableConvTranspose2d
from .lstm import StreamableLSTM


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                             causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetResnetBlock2d(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv2d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv2d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                             causal=causal, pad_mode=pad_mode, padding=(1, 1))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(channels, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=block_norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv1d(mult * n_filters, mult * n_filters * 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            StreamableConv1d(mult * n_filters, dimension, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class EncoderBlock2d(nn.Module):
    def __init__(self, act, ratio, n_residual_layers, ch_in, ch_out, residual_kernel_size, dilation_base, block_norm, norm_params, activation, activation_params, causal, pad_mode, compress, true_skip):
        super().__init__()
        model = []
        for j in range(n_residual_layers):
            model += [
                SEANetResnetBlock2d(ch_in, kernel_sizes=[residual_kernel_size, 1],
                                    dilations=[dilation_base ** j, 1],
                                    norm=block_norm, norm_params=norm_params,
                                    activation=activation, activation_params=activation_params,
                                    causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv2d(ch_in, ch_out,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
        
        self.model = nn.Sequential(*model)


class SEANetEncoder2d(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, 
                 frequency_bins: int = 512, 
                 num_1x1_convolutions: int = 2,
                 temporal_ratios: tp.List[int] = [2]):
        super().__init__()
        self.num_1x1_convolutions = num_1x1_convolutions
        self.temporal_ratios = temporal_ratios
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.frequency_bins = frequency_bins
        assert self.frequency_bins == 512, "model handles 512 frequency channels, change manually and in config file" 
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)

        chnls = n_filters
        freq_bins = frequency_bins

        model: tp.List[nn.Module] = [
            StreamableConv2d(2, chnls, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Take strides accross frequency dimension
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock2d(chnls, kernel_sizes=[residual_kernel_size, 1],
                                        dilations=[dilation_base ** j, 1],
                                        norm=block_norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv2d(chnls, chnls * 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            chnls *= 2
            freq_bins //= 2

        # decrease channels using 1x1 convolutions
        for _ in range(self.num_1x1_convolutions):
            for _ in range(n_residual_layers):
                model += [
                    SEANetResnetBlock2d(chnls, kernel_sizes=[1, 1],
                                        dilations=[1, 1],
                                        norm=block_norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv2d(chnls, chnls // 2,
                                 kernel_size=1, stride=1,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            chnls //= 2

        # flatten: results in output shape of (B, chnsl * freq_bins, T)
        model += [nn.Flatten(1, 2)]
        flattened_dim = chnls * freq_bins

        # force desired latent size
        model += [
            act(**activation_params),
            StreamableConv1d(flattened_dim, dimension, 1,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        # take temporal strides
        if len(self.temporal_ratios) > 0:
            for i, ratio in enumerate(self.temporal_ratios):
                block_norm = 'none' if self.disable_norm_outer_blocks >= i + 2 else norm
                # Add residual layers
                for j in range(n_residual_layers):
                    model += [
                        SEANetResnetBlock(dimension, kernel_sizes=[3, 1],
                                        dilations=[1, 1],
                                        norm=block_norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

                # Add downsampling layers
                model += [
                    act(**activation_params),
                    StreamableConv1d(dimension,dimension,
                                    kernel_size=ratio * 2, stride=ratio,
                                    norm=block_norm, norm_kwargs=norm_params,
                                    causal=causal, pad_mode=pad_mode),
                ]

        if lstm:
            model += [StreamableLSTM(dimension, num_layers=lstm)]

        model += [
            act(**activation_params),
            StreamableConv1d(dimension, dimension, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                          kernel_size=ratio * 2, stride=ratio,
                                          norm=block_norm, norm_kwargs=norm_params,
                                          causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=block_norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(n_filters, channels, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y


class SEANetDecoder2d(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0, frequency_bins: int = 512, 
                 num_1x1_convolutions: int = 2,
                 temporal_ratios: tp.List[int] = [2]):
        super().__init__()
        self.num_1x1_convolutions = num_1x1_convolutions
        self.temporal_ratios = temporal_ratios
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.frequency_bins = frequency_bins
        assert self.frequency_bins == 512, "model handles 512 frequency channels, change manually and in config file" 
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, dimension, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
                

        if lstm:
            model += [StreamableLSTM(dimension, num_layers=lstm)]

        # reverse temporal strides 
        if len(temporal_ratios) > 0:
            for i, ratio in enumerate(self.temporal_ratios):
                block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
                # Add upsampling layers
                model += [
                    act(**activation_params),
                    StreamableConvTranspose1d(dimension, dimension,
                                            kernel_size=ratio * 2, stride=ratio,
                                            norm=block_norm, norm_kwargs=norm_params,
                                            causal=causal, trim_right_ratio=trim_right_ratio),
                ]
                # Add residual layers
                for _ in range(n_residual_layers):
                    model += [
                        SEANetResnetBlock(dimension, kernel_sizes=[3, 1],
                                        dilations=[1, 1],
                                        activation=activation, activation_params=activation_params,
                                        norm=block_norm, norm_params=norm_params, causal=causal,
                                        pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

        chnls = n_filters * (2 ** (len(self.ratios) - self.num_1x1_convolutions))
        freq_bins = frequency_bins // np.product(self.ratios)
        

        model += [
            StreamableConv1d(dimension, chnls * freq_bins, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        model += [nn.Unflatten(1, (int(chnls), int(freq_bins)))]

        # scale channels using 1x1 convolutions
        for _ in range(self.num_1x1_convolutions):
            for _ in range(n_residual_layers):
                model += [
                    SEANetResnetBlock2d(chnls, kernel_sizes=[1, 1],
                                        dilations=[1, 1],
                                        norm=block_norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv2d(chnls, chnls * 2,
                                 kernel_size=1, stride=1,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            chnls *= 2

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose2d(chnls, chnls // 2,
                                          kernel_size=ratio * 2, stride=ratio,
                                          norm=block_norm, norm_kwargs=norm_params,
                                          causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock2d(chnls // 2, kernel_sizes=[residual_kernel_size, 1],
                                        dilations=[dilation_base ** j, 1],
                                        activation=activation, activation_params=activation_params,
                                        norm=block_norm, norm_params=norm_params, causal=causal,
                                        pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            chnls //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv2d(n_filters, 2, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y
