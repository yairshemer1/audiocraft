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
                    exclude='learnfair[0234,0300,0301,0302,0303,0304,0305,0306,0307,0308,0309,0310,0311,0312,0313,0314,0315,0316,0317,0318,0319,0320,0321,0322,0323,0324,0325,0326,0327,0328,0329,0330,0331,0332,0333,0334,0335,0336,0337,0338,0339,0340,0341,0342,0343,0344,0345,0346,0347,0348,0349,0350,0351,0352,0353,0354,0355,0356,0357,0358,0359,0360,0361,0362,0363,0364,0365,0366,0367,0368,0369,0370,0371,0372,0373,0374,0375,0376,0377,0378,0379,0380,0381,0382,0383,0384,0385,0386,0387,0388,0389,0390,0391,0392,0573,0597,0705,0799,2085,2236,2427,5197,5201,5203,5205,0825,0877,0861,0862,0863,0864,0873,0874,7572,7573,7574,7604,7606,7605,7516,7518,7519,7641,7642,7643,2369,2370,2371,2372,7524,7527,5130]')

    # base causal EnCodec trained on monophonic audio sampled at 32 kHz
    launcher.bind_({
        'solver': 'compression/complex_reconstruct',
        'logging.log_wandb': False,
        'checkpoint.save_every': None,
        'checkpoint.keep_last': 1,
        'dataset.batch_size': 64,
        'rvq.bins': 1024,
    })

    # dsets = ["audio/valentini_56spk", "audio/valentini_noisy_56spk"]
    model_dset_sr_name = [
        # ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 8000, 'denoise-ours,single task 8-8'),
        # ("encodec/complex/super_res", "audio/valentini_56spk", 16000, 'sr-ours,single task 8-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla 16-16'),
        # ("encodec/complex/vanilla", "audio/valentini_56spk", 8000, 'compress-ours vanilla 8-8'),
        ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk", 16000, 'compress-ours sr+denoise 8-16'),
    ]
    model_dset_sr_name_encodec = [
        # ("encodec/encodec_large_nq4_s320", "audio/valentini_56spk", 16000, 'compress-encodec 16-16'),
        # ("encodec/encodec_large_nq4_s320", "audio/valentini_56spk", 8000, 'compress-encodec 8-8'),

        # ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk"),
    ]

    model_dset_sr_name_temp = [
        # ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x2 16-16', [2]),
        # ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x4 16-16', [2,2])
    ]

    model_dset_sr_name_freq_hparams = [
        # ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x2 16-16', 2047, 500),
        # ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, 'compress-ours vanilla, temporal x4 16-16', 4095, 1000)
    ]

    model_dset_sr_name_encodec1d = [
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 16000, '1024 bins; new-denoise-1D-ours,denoise 16-16'),
        ("encodec/complex_1d/1d_sr", "audio/valentini_56spk", 16000, '1024 bins; new-sr-1D-ours,sr 8-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, '1024 bins; compress-1D-ours vanilla 16-16'),
        ("encodec/complex_1d/1d_sr_denoise", "audio/valentini_noisy_56spk", 16000, '1024 bins; compress-1D-ours sr + denoise 8-16'),
    ]

    model_dset_sr_name_encodec1d_ratios = [
        # ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, 'compress-x2-1D-ours vanilla 16-16', [2,1,1,1,1]),
        # ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, 'compress-x4-1D-ours vanilla 16-16', [2,2,1,1,1]),
    ]

    # launch xp
    with launcher.job_array():
        for flag in [False, True]:
            if flag:
                launcher.bind_({})

        # 1d models
        for model, dset, sr, label in model_dset_sr_name_encodec1d:
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
            }
            launcher(attrs)
            
        for model, dset, sr, label in model_dset_sr_name_encodec1d:
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
                'data_preprocess.hop_length': 320
            }
            launcher(attrs)

        # for model, dset, sr, label, ratios in model_dset_sr_name_encodec1d_ratios:

        #     attrs = {
        #         "model": model,
        #         "dset": dset,
        #         "sample_rate": sr,
        #         "seanet.ratios": ratios,
        #         "label": label,
        #     }
        #     launcher(attrs)

        # 2d models
        for model, dset, sr, label in model_dset_sr_name:
            label = "50Hz " + label
            attrs = {
                "model": model,
                "dset": dset,
                "sample_rate": sr,
                "label": label,
                'data_preprocess.hop_length': 320
            }
            launcher(attrs)

        # for model, dset, sr, label, temporal_ratios in model_dset_sr_name_temp:
        #     label = "50Hz " + label if flag else label
        #     attrs = {
        #         "model": model,
        #         "dset": dset,
        #         "sample_rate": sr,
        #         "label": label,
        #         "seanet.temporal_ratios": temporal_ratios,
        #     }
        #     launcher(attrs)

        # for model, dset, sr, label, nfft, hop in model_dset_sr_name_freq_hparams:
        #     label = "50Hz " + label if flag else label
        #     attrs = {
        #         "model": model,
        #         "dset": dset,
        #         "sample_rate": sr,
        #         "label": label,
        #         "data_preprocess.n_fft": nfft,
        #         "data_preprocess.win_length": nfft,
        #         "data_preprocess.hop_length": hop,
        #         "seanet.frequency_bins": int(np.ceil(nfft/2)),
        #     }
        #     launcher(attrs)

        # for model, dset, sr, label in model_dset_sr_name_encodec:
        #     label = "50Hz " + label if flag else label
        #     attrs = {
        #         "solver": "compression/reconstruct_encodec",
        #         "evaluate.every": 100000000,
        #         "generate.every": 100000000,
        #         "model": model,
        #         "dset": dset,
        #         "sample_rate": sr,
        #         "label": label,
        #     }
        #     launcher(attrs)
            