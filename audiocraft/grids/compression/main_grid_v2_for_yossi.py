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
                    # partition='scavenge',
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
        'dataset.batch_size': 32,
        # 'rvq.bins': 1024,
    })

    # encodec with 5 RVQs, add: rvq.n_q: 5
    # encodec_2_8kbs_model_dset_sr_name = [
    #     ("encodec/encodec_large_nq4_s320", "audio/valentini_56spk", 16000, 'n_q=5: compress-encodec 16-16'),
    #     ("encodec/encodec_large_nq4_s320", "audio/valentini_56spk", 8000, 'n_q=5: compress-encodec 8-8'),
    # ]

    # run our model - w.o quantization. add: encodec.quantizer: 'no_quant'
    # no_quant_model_dset_sr_name = [
    #     ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk", 16000, 'no_quant: compress-ours sr+denoise 8-16'),
    #     ("encodec/complex_1d/1d_sr_denoise", "audio/valentini_noisy_56spk", 16000, 'no_quant: compress-1D-ours sr + denoise 8-16')
    # ]

    # experiment with 256 codes and 4 cbs
    v1_model_dset_sr_name = [
        # 2d
        ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk", 16000, '256 codes, 4cbs - compress-ours sr+denoise 8-16'),
        ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 16000, '256 codes, 4cbs - denoise-ours,single task 16-16'),
        ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 8000, '256 codes, 4cbs - denoise-ours,single task 8-8'),
        ("encodec/complex/super_res", "audio/valentini_56spk", 16000, '256 codes, 4cbs - sr-ours,single task 8-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, '256 codes, 4cbs - compress-ours vanilla 16-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 8000, '256 codes, 4cbs - compress-ours vanilla 8-8'),
        # 1d
        ("encodec/complex_1d/1d_sr_denoise", "audio/valentini_noisy_56spk", 16000, '256 codes, 4cbs - compress-1D-ours sr + denoise 8-16'),
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 16000, '256 codes, 4cbs - new-denoise-1D-ours,denoise 16-16'),
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 8000, '256 codes, 4cbs - new-denoise-1D-ours,denoise 8-8'),
        ("encodec/complex_1d/1d_sr", "audio/valentini_56spk", 16000, '256 codes, 4cbs - new-sr-1D-ours,sr 8-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, '256 codes, 4cbs -  compress-1D-ours vanilla 16-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 8000, '256 codes, 4cbs -  compress-1D-ours vanilla 8-8'),
    ]
    
    # experiment with 2048 codes and 3 cbs
    v2_model_dset_sr_name = [
        # 2d
        ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk", 16000, '2048 codes, 3 cbs - compress-ours sr+denoise 8-16'),
        ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 16000, '2048 codes, 3 cbs - denoise-ours,single task 16-16'),
        ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 8000, '2048 codes, 3 cbs - denoise-ours,single task 8-8'),
        ("encodec/complex/super_res", "audio/valentini_56spk", 16000, '2048 codes, 3 cbs - sr-ours,single task 8-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, '2048 codes, 3 cbs - compress-ours vanilla 16-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 8000, '2048 codes, 3 cbs - compress-ours vanilla 8-8'),
        # 1d
        ("encodec/complex_1d/1d_sr_denoise", "audio/valentini_noisy_56spk", 16000, '2048 codes, 3 cbs - compress-1D-ours sr + denoise 8-16'),
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 16000, '2048 codes, 3 cbs - new-denoise-1D-ours,denoise 16-16'),
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 8000, '2048 codes, 3 cbs - new-denoise-1D-ours,denoise 8-8'),
        ("encodec/complex_1d/1d_sr", "audio/valentini_56spk", 16000, '2048 codes, 3 cbs - new-sr-1D-ours,sr 8-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, '2048 codes, 3 cbs -  compress-1D-ours vanilla 16-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 8000, '2048 codes, 3 cbs -  compress-1D-ours vanilla 8-8'),
    ]

    # experiment with 512 codes and 4 cbs
    v3_model_dset_sr_name = [
        # 2d
        ("encodec/complex/super_res_denoise", "audio/valentini_noisy_56spk", 16000, '512 codes, 4cbs - compress-ours sr+denoise 8-16'),
        ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 16000, '512 codes, 4cbs - denoise-ours,single task 16-16'),
        ("encodec/complex/denoise", "audio/valentini_noisy_56spk", 8000, '512 codes, 4cbs - denoise-ours,single task 8-8'),
        ("encodec/complex/super_res", "audio/valentini_56spk", 16000, '512 codes, 4cbs - sr-ours,single task 8-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 16000, '512 codes, 4cbs - compress-ours vanilla 16-16'),
        ("encodec/complex/vanilla", "audio/valentini_56spk", 8000, '512 codes, 4cbs - compress-ours vanilla 8-8'),
        # 1d
        ("encodec/complex_1d/1d_sr_denoise", "audio/valentini_noisy_56spk", 16000, '512 codes, 4cbs - compress-1D-ours sr + denoise 8-16'),
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 16000, '512 codes, 4cbs - new-denoise-1D-ours,denoise 16-16'),
        ("encodec/complex_1d/1d_denoise", "audio/valentini_noisy_56spk", 8000, '512 codes, 4cbs - new-denoise-1D-ours,denoise 8-8'),
        ("encodec/complex_1d/1d_sr", "audio/valentini_56spk", 16000, '512 codes, 4cbs - new-sr-1D-ours,sr 8-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 16000, '512 codes, 4cbs -  compress-1D-ours vanilla 16-16'),
        ("encodec/complex_1d/1d_vanilla", "audio/valentini_56spk", 8000, '512 codes, 4cbs -  compress-1D-ours vanilla 8-8'),
    ]

    # launch xp
    with launcher.job_array():

        # for model, dset, sr, name in encodec_2_8kbs_model_dset_sr_name:
        #     args = {
        #         "solver": "compression/reconstruct_encodec",
        #         "model": model,
        #         'dset': dset,
        #         "sample_rate": sr,
        #         "rvq.n_q": 5,
        #         'label': name
        #     }
        #     launcher(args)
            
        # for model, dset, sr, name in no_quant_model_dset_sr_name:
        #     args = {
        #         "solver": 'compression/complex_reconstruct',
        #         "model": model,
        #         'dset': dset,
        #         "sample_rate": sr,
        #         "encodec.quantizer": 'no_quant',
        #         'label': name
        #     }
        #     launcher(args)
                
        for model, dset, sr, name in v1_model_dset_sr_name:
            args = {
                "solver": 'compression/complex_reconstruct',
                "model": model,
                'dset': dset,
                "sample_rate": sr,
                'rvq.bins': 256,
                'label': name
            }
            launcher(args)       
        for model, dset, sr, name in v2_model_dset_sr_name:
            args = {
                "solver": 'compression/complex_reconstruct',
                "model": model,
                'dset': dset,
                "sample_rate": sr,
                'rvq.bins': 2048,
                'rvq.n_q': 3,
                'label': name
            }
            launcher(args)     
        for model, dset, sr, name in v3_model_dset_sr_name:
            args = {
                "solver": 'compression/complex_reconstruct',
                "model": model,
                'dset': dset,
                "sample_rate": sr,
                'rvq.bins': 512,
                'rvq.n_q': 4,
                'label': name
            }
            launcher(args)
