from torchvision import transforms
from torchvision.transforms import (
    RandomRotation,
    RandomCrop,
    Pad,
    Resize,
    RandomAffine,
    ToTensor,
    Compose,
    RandomPerspective,
    Grayscale,
    RandomApply,
)
from PIL import Image


def get_transform_train(upsample_size, final_size, channels=1):

    random_transform = []
    transform1 = transforms.Compose(
        [
            RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
            RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
            RandomPerspective(distortion_scale=0.3, p=0.1),
        ]
    )

    transform2 = transforms.Compose(
        [
            RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
            RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
        ]
    )

    transform_list = [transform1, transform2]

    transform_prob = 1.0 / len(transform_list)
    for transform in transform_list:
        random_transform.append(RandomApply([transform], transform_prob))

    transform_simple = Compose(
        [Resize(final_size), Grayscale(num_output_channels=1), ToTensor(),]
    )

    if channels == 3:
        transform_train = Compose(
            [
                RandomCrop(128),
                # Pad((0, 0, 1, 1), fill=0),
                Resize(upsample_size),
                RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
                Resize(final_size),
                ToTensor(),
            ]
        )
        return transform_train

    transform_train = Compose(
        [
            RandomCrop(128),
            # Pad((0, 0, 1, 1), fill=0),
            Resize(upsample_size),
            RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
            Resize(final_size),
            Grayscale(num_output_channels=1),
            ToTensor(),
        ]
    )

    return transform_train


def get_transform_test(final_size, channels=1):
    if channels == 3:
        transform_test = Compose([Resize(final_size), ToTensor(),])
        return transform_test

    transform_test = Compose(
        [Resize(final_size), Grayscale(num_output_channels=1), ToTensor(),]
    )

    return transform_test
