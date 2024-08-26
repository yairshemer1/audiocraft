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
        'logging.log_wandb': True,
        'checkpoint.save_every': None,
        'checkpoint.keep_last': 1,
        'dataset.batch_size': 32,
        'optim.updates_per_epoch': 5000,
        'optim.epochs': 100,
        'evaluate.every': 1000000,
        'generate.every': 1000000,
        # 'rvq.bins': 1024,
    })
    
    for model in ['dn_comp', 'sr_dn_comp']:
        dset = 'audio/valentini_noisy_56spk' if 'dn' in model else 'audio/valentini_clean_56spk'
        for bins, vqs, channels in [
            (2048, [1, 1, 1, 1], [1024, 1024, 1024, 512]),  # sanity for yair
            (512, [1, 1, 1, 2], [1024, 1024, 1024, 512]),
            (2048, [1, 1, 2], [1024, 1024, 1024]),
            (512, [1, 2, 2], [1024, 1024, 1024]),
        ]:
            args = {
                'dset': dset,
                'model': f"encodec/skips_1d/{model}",
                'model_config.channels': channels,
                'model_config.strides': [1] * len(vqs),
                'model_config.vqs.n_qs': vqs,
                'model_config.vqs.additional_kwargs.bins': bins,
                'wandb.name': f"{model}_{bins}_{vqs}_{channels}"
                }
            launcher(args)