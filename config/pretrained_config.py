import math

PRETRAINED_CONFIG = {
    "network_type": "CCT",
    "pretrained": False,
    "image_size": 224,
    "batch_size": 8,
    "num_epochs": 3,
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
        "img_size": (224, 224),
        "embedding_dim": 128,
        "n_conv_layers": 2,
        "kernel_size": 7,
        "stride": 2,
        "padding": 3,
        "pooling_kernel_size": 3,
        "pooling_stride" : 2,
        "pooling_padding" : 1,
        "num_layers": 5,
        "num_heads": 2,
        "mlp_radio": 2.0,
        "num_classes": 3,
        "positional_embedding": "learnable",  # ['sine', 'learnable', 'none']
        "n_input_channels": 1,
    },
}
