"""temporary infer"""

from __future__ import print_function
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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
)
from sklearn.metrics import roc_curve, auc, confusion_matrix
from utils.dataset import DeepLenseDataset


class Inference(object):
    def __init__(
        self,
        best_model,
        test_loader,
        device,
        num_classes,
        testset,
        dataset_name,
        labels_map,
        image_size,
        channels,
        log_dir,
        destination_dir="data",
    ) -> None:

        self.best_model = best_model
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.testset = testset
        self.destination_dir = destination_dir
        self.dataset_name = dataset_name
        self.labels_map = labels_map
        self.image_size = image_size
        self.channels = channels
        self.log_dir = log_dir

    def to_one_hot_vector(self, label):
        b = np.zeros((label.shape[0], self.num_classes))
        b[np.arange(label.shape[0]), label] = 1

        return b.astype(int)

    def infer_plot_roc(self):
        total = 0
        all_test_loss = []
        all_test_accuracy = []
        label_true_arr = []
        label_true_arr_onehot = []
        label_pred_arr = []
        pred_arr = []
        plt.rcParams.update(plt.rcParamsDefault)
        fig = plt.figure()

        correct = 0
        with torch.no_grad():
            self.best_model.eval()
            for i, (x, t) in enumerate(self.test_loader):
                x = x.to(self.device)
                t = t.to(self.device)
                y = self.best_model(x)

                pred_arr.append(y.cpu().numpy())

                _, prediction = torch.max(y.data, 1)
                label_pred_arr.append(prediction.cpu().numpy())
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                label_true_arr.append(t.cpu().numpy())

                one_hot_t = self.to_one_hot_vector(t.cpu().numpy())
                label_true_arr_onehot.append(one_hot_t)

        self.y_pred = []
        for i in label_pred_arr:
            for j in i:
                self.y_pred.append(j)
        self.y_pred = np.array(self.y_pred)

        y_true_onehot = []
        for i in label_true_arr_onehot:
            for j in i:
                y_true_onehot.append(list(j))
        y_true_onehot = np.array(y_true_onehot)

        y_score = []
        for i in pred_arr:
            for j in i:
                y_score.append(list(j))
        y_score = np.array(y_score)

        self.y_true = []
        for i in label_true_arr:
            for j in i:
                self.y_true.append(j)
        self.y_true = np.array(self.y_true)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        self.inv_map = self.labels_map  # {v: k for k, v in self.labels_map.items()}

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_onehot.ravel(), y_score.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.4f})"
            "".format(roc_auc["micro"]),
        )
        for i in range(self.num_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label="ROC curve of class " + self.inv_map[i] + " (area = {0:0.4f})"
                "".format(roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Transformer ROC")
        plt.legend(loc="lower right")
        plt.savefig(f"{self.log_dir}/roc.png", dpi=150)
        plt.show()
        # fig.savefig(f"{self.log_dir}/roc.png", dpi=150)

    def plot_confusion_matrix(
        self, cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/confusion_matrix.png", dpi=150)
        plt.show()

    def generate_plot_confusion_matrix(self):
        cnf_matrix = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1, 2])

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(
            cnf_matrix,
            classes=[self.inv_map[0], self.inv_map[1], self.inv_map[2]],
            title="Confusion matrix",
        )

    def rot_equivariance(
        self,
        model: torch.nn.Module,
        x,
        device,
        labels_map,
        resize1,
        resize2,
        pad,
        to_tensor,
        to_gray,
        image_size,
        channels,
    ):
        # evaluate the `model` on 8 rotated versions of the input image `x`
        model.eval()

        wrmup = model(torch.randn(1, channels, image_size, image_size).to(device))
        del wrmup

        x = resize1(pad(x))

        print("###########################")
        header = "angle |  " + "  ".join(
            ["{}".format(value) for key, value in labels_map.items()]
        )
        print(header)
        with torch.no_grad():
            for r in range(8):
                x_transformed = to_tensor(
                    to_gray(resize2(x.rotate(r * 45.0, Image.BILINEAR)))
                )  # .reshape(1, 1, 29, 29)
                x_transformed = x_transformed.unsqueeze(0).to(device)

                y = model(x_transformed)
                y = y.to("cpu").numpy().squeeze()

                angle = r * 45
                print("{:5d} : {}".format(angle, y))
        print("###########################")

    def test_equivariance(self):
        valset_notransform = DeepLenseDataset(
            self.destination_dir, "test", self.dataset_name, transform=None
        )
        x, y = next(iter(valset_notransform))
        print(self.labels_map[y])
        self.rot_equivariance(
            model=self.best_model,
            x=x,
            device=self.device,
            labels_map=self.labels_map,
            resize1=Resize(387),
            resize2=Resize(self.image_size),
            pad=Pad((0, 0, 1, 1), fill=0),
            to_tensor=ToTensor(),
            to_gray=Grayscale(num_output_channels=1),
            image_size=self.image_size,
            channels=self.channels,
        )

