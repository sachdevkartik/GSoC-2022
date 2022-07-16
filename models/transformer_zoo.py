from .crossformer import CrossFormer
from .twins_svt import TwinsSVT
from .levit import LeViT
from .pit import PiT


def GetCrossFormer(num_classes, num_channels):
    model = CrossFormer(
        num_classes=num_classes,  # number of output classes
        channels=num_channels,
        dim=(32, 64, 128, 256),  # dimension at each stage
        depth=(2, 2, 4, 2),  # depth of transformer at each stage
        global_window_size=(8, 4, 2, 1),  # global window sizes at each stage
        local_window_size=7,  # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
    )
    return model


def GetTwinsSVT(num_classes, num_channels):
    model = TwinsSVT(
        num_classes=num_classes,  # number of output classes
        s1_emb_dim=16,  # stage 1 - patch embedding projected dimension
        s1_patch_size=4,  # stage 1 - patch size for patch embedding
        s1_local_patch_size=7,  # stage 1 - patch size for local attention
        s1_global_k=7,  # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
        s1_depth=1,  # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
        s2_emb_dim=16,  # stage 2 (same as above)
        s2_patch_size=2,
        s2_local_patch_size=7,
        s2_global_k=7,
        s2_depth=2,
        s3_emb_dim=16,  # stage 3 (same as above)
        s3_patch_size=2,
        s3_local_patch_size=7,
        s3_global_k=7,
        s3_depth=3,
        s4_emb_dim=16,  # stage 4 (same as above)
        s4_patch_size=2,
        s4_local_patch_size=7,
        s4_global_k=7,
        s4_depth=1,
        peg_kernel_size=3,  # positional encoding generator kernel size
        dropout=0.0,  # dropout
        channels=num_channels,
        heads=5,
    )
    return model


def GetLeViT(num_classes, num_channels, img_size):

    model = LeViT(
        image_size=img_size,
        num_classes=num_classes,
        stages=3,  # number of stages
        dim=(64, 128, 128),  # dimensions at each stage
        depth=5,  # transformer of depth 4 at each stage
        heads=(2, 4, 5),  # heads at each stage
        mlp_mult=2,
        dropout=0.1,
        channels=num_channels,
    )
    return model


def GetPiT(num_classes, num_channels, img_size):

    model = PiT(
        image_size=img_size,
        patch_size=14,
        dim=128,
        num_classes=num_classes,
        depth=(
            3,
            3,
            3,
        ),  # list of depths, indicating the number of rounds of each stage before a downsample
        heads=1,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        channels=num_channels,
    )
    return model
