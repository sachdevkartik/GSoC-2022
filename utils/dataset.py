import os
import gdown
import splitfolders
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from config.data_config import DATASET
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage


def download_dataset(
    filename: str,
    url: str = "https://drive.google.com/uc?id=1m7QzSzXyE8u_QoYplN9dIe-X2pf1KXxt",
) -> None:

    if not os.path.isfile(filename):
        try:
            gdown.download(url, filename, quiet=False)
        except Exception as e:
            print(e)
    else:
        print("File exists")


def extract_split_dataset(  # change to filename
    filename: str,
    destination_dir: str = "data",
    dataset_name: str = "Model_I",
    split: bool = False,
) -> None:

    # only extracting folder
    if not split:
        print("Extracting folder ...")
        os.system(f"tar xf {filename} --directory {destination_dir}")
        print("Extraction complete")
        # os.system(f"rm -r {filename}")

    # splitting folder
    else:
        os.system(
            f"tar xf {filename} --directory {destination_dir} ; mv {destination_dir}/{dataset_name} {destination_dir}/{dataset_name}_raw"
        )
        splitfolders.ratio(
            f"{destination_dir}/{dataset_name}_raw",
            output=f"{destination_dir}/{dataset_name}",
            seed=1337,
            ratio=(0.9, 0.1),
        )
        os.system(f"rm -r {destination_dir}/{dataset_name}_raw")


class DeepLenseDataset(Dataset):
    # TODO: add val-loader + splitting
    def __init__(
        self,
        destination_dir,
        mode,
        dataset_name,
        transform=None,
        download="False",
        channels=1,
    ):
        assert mode in ["train", "test", "val"]

        if mode == "train":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/train"

        elif mode == "val":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/val"

        else:
            filename = f"{destination_dir}/{dataset_name}_test.tgz"
            foldername = f"{destination_dir}/{dataset_name}_test"
            # self.root_dir = foldername

        url = DATASET[f"{dataset_name}"][f"{mode}_url"]

        if download and not os.path.isdir(foldername) is True:
            if not os.path.isfile(filename):
                download_dataset(
                    filename, url=url,
                )
            extract_split_dataset(filename, destination_dir)
        else:
            assert (
                os.path.isdir(foldername) is True
            ), "Dataset doesn't exists, set arg download to True!"

            print("Dataset already exists")

        self.root_dir = foldername

        self.transform = transform
        classes = os.listdir(
            self.root_dir
        )  # [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.imagefilename = []
        self.labels = []
        self.channels = channels

        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename.append(os.path.join(self.root_dir, i, x))
                self.labels.append(self.class_to_idx[i])

    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]

        image = np.load(image, allow_pickle=True)  # , mmap_mode='r'
        if label == 0:
            image = image[0]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=2)

        # image = image / image.max()  # normalizes data in range 0 - 255
        # image = 255 * image
        # image = (image - np.min(image))/(np.max(image) - np.min(image))
        # if self.channels == 3:
        #     image = Image.fromarray(image.astype("uint8")).convert("RGB")
        # else:
        #     image = Image.fromarray(image.astype("uint8"))  # .convert("RGB")

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            image = (
                image.float().clone().detach()  # .requires_grad_(True)
            )  # torch.tensor(image, dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.labels)


def visualize_samples(
    dataset, labels_map, fig_height=15, fig_width=15, num_cols=5, cols_rows=5
):  # trainset
    # labels_map = {0: "axion", 1: "cdm", 2: "no_sub"}
    figure = plt.figure(figsize=(fig_height, fig_width))
    cols, rows = num_cols, cols_rows
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"{labels_map[label]}")
        plt.axis("off")
        im = ToPILImage()(img)
        img = img.squeeze()
        plt.imshow(img, cmap="gray")
        # plt.imshow(img)
    plt.show()


class DeepLenseDataset_dataeff(Dataset):
    # TODO: add val-loader + splitting
    def __init__(
        self,
        destination_dir,
        mode,
        dataset_name,
        transform=None,
        download="False",
        channels=1,
    ):
        assert mode in ["train", "test", "val"]

        if mode == "train":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/train"

        elif mode == "val":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/val"

        else:
            filename = f"{destination_dir}/{dataset_name}_test.tgz"
            foldername = f"{destination_dir}/{dataset_name}_test"
            # self.root_dir = foldername

        url = DATASET[f"{dataset_name}"][f"{mode}_url"]

        if download and not os.path.isdir(foldername) is True:
            if not os.path.isfile(filename):
                download_dataset(
                    filename, url=url,
                )
            extract_split_dataset(filename, destination_dir)
        else:
            assert (
                os.path.isdir(foldername) is True
            ), "Dataset doesn't exists, set arg download to True!"

            print("Dataset already exists")

        self.root_dir = foldername

        self.transform = transform
        classes = os.listdir(
            self.root_dir
        )  # [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # self.imagefilename = []
        self.labels = []
        self.channels = channels

        # TODO: make dynamic
        if mode == "train":
            num = 87525
        elif mode == "test":
            num = 15000

        # if not os.path.exists(f"images_mmep_{mode}.npy"):

        self.images_mmep = np.memmap(
            f"images_mmep_{mode}.npy", dtype="int16", mode="w+", shape=(num, 150, 150),
        )

        self.labels_mmep = np.memmap(
            f"labels_mmep_{mode}.npy", dtype="float64", mode="w+", shape=(num, 1)
        )

        w_index = 0
        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename = os.path.join(self.root_dir, i, x)
                image = np.load(self.imagefilename, allow_pickle=True)
                label = self.class_to_idx[i]
                if label == 0:
                    image = image[0]
                self.images_mmep[w_index, :] = image
                self.labels_mmep[w_index] = label
                self.labels.append(self.class_to_idx[i])
                w_index += 1

    def __getitem__(self, index):
        image = np.asarray(self.images_mmep[index])
        label = np.asarray(self.labels_mmep[index], dtype="int64")[0]

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        if self.channels == 3:
            image = Image.fromarray(image.astype("uint8")).convert("RGB")
        else:
            image = Image.fromarray(image.astype("uint8"))  # .convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)
