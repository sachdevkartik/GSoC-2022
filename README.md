# __GSoC-2022__
Solution for evaluation tests for GSoC DeepLense 2022. <br> 
This repo solves the following tests:

<br>

## __Common Test I. Multi-Class Classification__ ##
An Equivariant neural network is built using PyTorch for classifying the images into lenses using PyTorch. Approach and strategy are discussed in [test1.ipynb](./test1.ipynb)

**Dataset**: The Dataset consists of three classes, strong lensing images with no substructure, subhalo substructure, and vortex substructure. 

**Solution**: The notebook can be open on [GoogleColab](https://colab.research.google.com/github/sachdevkartik/GSoC-2022/blob/main/test1.ipynb)


**Model Weights**: [cnn_state_dict_model.pt](model/cnn_state_dict_model.pt)

**Inference**: Please use [test1_inference.ipynb](./test1_inference.ipynb) file for inference.


<br>

## __Specific Test V. Exploring Transformers__ ##

### __Approach 1__ ##

An efficient Convolutional Vision Transformer (CvT) is built for binary classification using PyTorch. Approach and strategy are discussed in [test2.ipynb](./test2.ipynb)

**Dataset**: The Dataset consists of simulated strong gravitational lensing images with and without substructure. 

**Solution**: The notebook can be open on [GoogleColab](https://colab.research.google.com/github/sachdevkartik/GSoC-2022/blob/main/test2.ipynb)

**Model Weights**: [ConvTransformer_2022-03-15-13-40-54.pt](model/ConvTransformer_2022-03-15-13-40-54.pt)

**Inference**: Please use [test2_inference.ipynb](./test2_inference.ipynb) file for inference.

<br>

### __Approach 2__ ##

A lightweight Distillable Vision Transformer is built for binary classification using PyTorch. Approach and strategy are discussed in [test2_DistillableViT.ipynb](./test2_DistillableViT.ipynb)

**Dataset**: The Dataset consists of simulated strong gravitational lensing images with and without substructure. 

**Solution**: The notebook can be open on [GoogleColab](https://colab.research.google.com/github/sachdevkartik/GSoC-2022/blob/main/test2_DistillableViT.ipynb)

**Model Weights**: [DistillableViT_2022-03-19-00-18-42.pt](model/DistillableViT_2022-03-19-00-18-42.pt)

**Inference**: Please use [test2_DistillableViT_inference.ipynb](./test2_DistillableViT_inference.ipynb) file for inference.

<br>


## __Locally running__
To run the notebooks locally and install required dependencies, please execute the following:
 ```
 bash setup.bash
  ```
<br>

## __Citation__


* [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251)
 
    ```
    @inproceedings{e2cnn,
        title={{General E(2)-Equivariant Steerable CNNs}},
        author={Weiler, Maurice and Cesa, Gabriele},
        booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
        year={2019},
    }
    ```
* [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808)


  ```
  @article{wu2021cvt,
    title={Cvt: Introducing convolutions to vision transformers},
    author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
    journal={arXiv preprint arXiv:2103.15808},
    year={2021}
  }
  ```
* [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)


    ```
    @article{touvron2020deit,
      title={Training data-efficient image transformers & distillation through attention},
      author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
      journal={arXiv preprint arXiv:2012.12877},
      year={2020}
    }
    ```

* Apoorva Singh, Yurii Halychanskyi, Marcos Tidball, DeepLense, (2021), GitHub repository, https://github.com/ML4SCI/DeepLense


