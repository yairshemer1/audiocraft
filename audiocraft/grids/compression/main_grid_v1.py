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

from ..compression._explorers import CompressionExplorer


@CompressionExplorer
def explorer(launcher):

    # n_gpus = 16
    n_gpus = 64

    launcher.slurm_(gpus=n_gpus, time=4320,
                    partition='devlab,learnlab,learnfair',
                    # constraint='ampere80gb',
                    constraint='volta32gb',
                    exclude='learnfair[5076,5072,5234,5201,5122,7591,1873,2404,7556,2455,5299,7505,5209,5246,5162,5048,5224,5186,5137,5059,5087,5071,2116,5233,5055,5053,5136,2294]')

    # base causal EnCodec trained on monophonic audio sampled at 32 kHz
    launcher.bind_({
        'solver': 'compression/complex_reconstruct',
        'model': 'h_encodec/acoustic_semantic_mfcc',
        'logging.log_wandb': True,
        'dataset.generate.num_samples': 10,
        'dataset.segment_duration': 5.0,
        'dataset.generate.segment_duration': 5.0,
        'optim.updates_per_epoch': 2500,
        'dataset.batch_size': 128,
        'dataset.generate.batch_size': 128,
        'dataset.evaluate.batch_size': 128,
        'optim.lr': 5e-5,
        'optim.optimizer': 'adamw',
    })

    # launch xp
    with launcher.job_array():

        # for cpc_sem_sem, detach_cond in [(True, True), (True, False)]:
        for detach_cpc_targets, detach_cond in [(False, True), (False, False), (True, True), (True, False)]:

            name = f"semantic hencodec; cpc detach_cpc_targets: {detach_cpc_targets}, detach cond: {detach_cond}"
            attrs = {
                "detach_cpc_targets": detach_cpc_targets,
                "additional_kwargs.detach_semantic_branch": detach_cond,
                'label': name,
                'wandb.name': name,
            }
            launcher(attrs)

