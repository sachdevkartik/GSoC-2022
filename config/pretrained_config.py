import math

PRETRAINED_CONFIG = {
    "network_type": "PiT",
    "pretrained": True,
    "image_size": 224,
    "batch_size": 32,
    "num_epochs": 15,
    "optimizer_config": {
        "name": "AdamW",
        "weight_decay": 0.01,
        "lr": 0.001,
        "momentum": 0.9,
        "betas": (0.9, 0.999),
        "warmup_epoch": 3,
    },
    "lr_schedule_config": {
        "use_lr_schedule": True,
        "step_lr": {"gamma": 0.5, "step_size": 20,},
        "reduce_on_plateau": {
            "factor": 0.1,
            "patience": 4,
            "threshold": 0.0000001,
            "verbose": True,
        },
    },
    "channels": 1,
    "network_config": {
        "s1_emb_dim": 32,  # stage 1 - (same as above)
        "s1_emb_kernel": 3,
        "s1_emb_stride": 2,
        "s1_proj_kernel": 3,
        "s1_kv_proj_stride": 2,
        "s1_heads": 2,
        "s1_depth": 2,
        "s1_mlp_mult": 2,
        "mlp_last": 64,
        "dropout": 0.1,
        "sym_group": "Circular",
        "N": 4,
        "e2cc_mult_1": 20,
    },
}
