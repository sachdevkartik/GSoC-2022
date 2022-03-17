# GSoC-2022
Solution for evaluation tests for GSoC 2022 
This repo solves the following tasks:

<br>

## __Common Test I. Multi-Class Classification__ ##
An Equivariant neural network is built using PyTorch for classifying the images into lenses using PyTorch. Approach and strategy are discussed in [test1.ipynb](./test1.ipynb)

**Dataset**: The Dataset consists of three classes, strong lensing images with no substructure, subhalo substructure, and vortex substructure. 

**Solution**: The notebook can be open on [GoogleColab](https://colab.research.google.com/github/sachdevkartik/GSoC-2022/blob/main/test1.ipynb)


**Model Weights**: 


<br>

## __Specific Test V. Exploring Transformers__ ##

An efficient Convolutional Vision Transformer (CvT) is built for binary classification using PyTorch. Approach and strategy are discussed in [test2.ipynb](./test2.ipynb)

**Dataset**: The Dataset consists of simulated strong gravitational lensing images with and without substructure. 

**Solution**: The notebook can be open on [GoogleColab](https://colab.research.google.com/github/sachdevkartik/GSoC-2022/blob/main/test2.ipynb)
 or run locally after executing ```setup.bash ```

**Model Weights**: 

<br>

 ## __Misc: Further exploring Transformers__ 
 To make more efficient and robust transformers, I have tried a couple of alternatives than vanilla and convolutional transformers.

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

* Apoorva Singh, Yurii Halychanskyi, Marcos Tidball, DeepLense, (2021), GitHub repository, https://github.com/ML4SCI/DeepLense


