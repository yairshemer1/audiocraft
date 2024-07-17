# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Grid search file, simply list all the exp you want in `explorer`.
Any new exp added there will be scheduled.
You can cancel and experiment by commenting its line.

This grid shows how to train a base causal EnCodec model at 24 kHz.
"""
import numpy as np
from ..compression._explorers import CompressionExplorer


@CompressionExplorer
def explorer(launcher):

    # n_gpus = 16
    n_gpus = 8

    launcher.slurm_(gpus=n_gpus, time=4320,
                    partition='devlab,learnlab,learnfair,scavenge',
                    # constraint='ampere80gb',
                    constraint='volta32gb',
                    exclude='learnfair[5076,5072,5234,5201,5122,7591,1873,2404,7556,2455,5299,7505,5209,5246,5162,5048,5224,5186,5137,5059,5087,5071,2116,5233,5055,5053,5136,2294]')

    # base causal EnCodec trained on monophonic audio sampled at 32 kHz
    launcher.bind_({
        'solver': 'compression/complex_reconstruct',
        'logging.log_wandb': False,
        'checkpoint.save_every': None,
        'checkpoint.keep_last': 1,
        'dataset.batch_size': 32,
    })

    # dsets = ["audio/valentini_56spk", "audio/valentini_noisy_56spk"]
    model_dset_sr_name = [
        ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 8000, 'denoise-ours,single task 8-8'),
        ("encodec/complex/super_res", "audio/valentini_56spk", 16000, 'sr-ours,single task 8-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla 16-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 8000, 'compress-ours vanilla 8-8'),

        # ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk"),
    ]
    model_dset_sr_name_encodec = [
        ("encodec/encodec_large_nq4_s320", "audio/valentini_56spk", 16000, 'compress-encodec 16-16'),
        ("encodec/encodec_large_nq4_s320", "audio/valentini_56spk", 8000, 'compress-encodec 8-8'),

        # ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk"),
    ]

    model_dset_sr_name_temp = [
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x2 16-16', [2]),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x4 16-16', [2,2])
    ]

    model_dset_sr_name_freq_hparams = [
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x2 16-16', 2047, 500),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x4 16-16', 4095, 1000)
    ]

    model_dset_sr_name_encodec1d = [
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 16000, 'denoise-1D-ours,denoise 16-16'),
        ("encodec/complex_1d/1d_sr", "audio/valentini_56spk", 16000, 'sr-1D-ours,sr 8-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, 'compress-1D-ours vanilla 16-16'),
        ("encodec/complex_1d/1d_sr_denoise", "audio/valentini_noisy_56spk", 16000, 'compress-1D-ours sr + denoise 8-16'),
    ]
    
    model_dset_sr_name_encodec1d_ratios = [
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, 'compress-x2-1D-ours vanilla 16-16', [2,1,1,1,1]),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, 'compress-x4-1D-ours vanilla 16-16', [2,2,1,1,1]),
    ]

    # launch xp
    with launcher.job_array():

        # 1d models
        for model, dset, sr, label in model_dset_sr_name_encodec1d:
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
            }
            launcher(attrs)

        for model, dset, sr, label,ratios in model_dset_sr_name_encodec1d_ratios:
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "seanet.ratios": ratios,
                "label": label,
            }
            launcher(attrs)

        # 2d models
        for model, dset, sr, label in model_dset_sr_name:
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
            }
            launcher(attrs)

        for model, dset, sr, label, temporal_ratios in model_dset_sr_name_temp:
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
                "seanet.temporal_ratios": temporal_ratios,
            }
            launcher(attrs)

        for model, dset, sr, label, nfft, hop in model_dset_sr_name_freq_hparams:
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
                "data_preprocess.n_fft": nfft,
                "data_preprocess.win_length": nfft,
                "data_preprocess.hop_length": hop,
                "seanet.frequency_bins": int(np.ceil(nfft/2)),
            }
            launcher(attrs)

        for model, dset, sr, label in model_dset_sr_name_encodec:
            attrs = {
                "solver": "compression/reconstruct_encodec",
                "evaluate.every": 100000000,
                "generate.every": 100000000,
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
            }
            launcher(attrs)