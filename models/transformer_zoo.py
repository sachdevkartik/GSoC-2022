from vit_pytorch.crossformer import CrossFormer


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
