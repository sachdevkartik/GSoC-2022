# __GSoC-2022__
 __Ongoing GSoC project__ <br>
<br>

# __Tested Transformer architectures__

<br>

### __[CvT](https://arxiv.org/pdf/2103.15808.pdf)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  91.54  | 0.9714  | 0.9657 | 0.9991 |
  | Model II  | 99.41  | 0.9987  | 0.9990 | 1.0000 |
  | Model III |  99.04   | 0.9986  | 0.9992 | 1.0000 |

<br>

### __[CrossFormer](https://arxiv.org/pdf/2203.13387.pdf)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  91.20  | 0.9508  | 0.9511 | 0.9996 |
  | Model II  | 97.42  | 0.9856  | 0.6041 | 0.9998 |
  | Model III |  98.13   | 0.9782  | 0.9096 | 0.9998 |

<br>

### __[LeViT](https://openaccess.thecvf.com/content/ICCV2021/papers/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.pdf)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   | 91.64   |  0.9720 | 0.8612 | 0.9997 |
  | Model II  | 97.12  | 0.9936  | 0.9800 | 1.0000 |
  | Model III | 97.97    | 0.9981  | 0.9973 | 1.0000 |

<br>

### __[TwinsSVT](https://arxiv.org/abs/2104.13840)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  91.08   | 0.6678  | 0.9179 | 0.9993 |
  | Model II  |   97.44  | 0.9812  | 0.6572 | 0.9996 |
  | Model III | 98.48    | 0.9942  | 0.5183 | 0.9999 |
  
<br>

### __[CCT](https://arxiv.org/abs/2104.05704v4)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  89.71  | 0.9632  | 0.9462 | 0.9988 |
  | Model II  |  69.68 | 0.9566  | 0.8216 | 0.9061 |
  | Model III | 99.48    | 0.9999  | 0.9998 | 1.0000 |


<br>

### __[CrossViT](https://arxiv.org/abs/2103.14899)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  84.20   | 0.9261  | 0.8816 | 0.9988 |
  | Model II  |  91.33    | 0.9816  | 0.9386 | 0.9959 |
  | Model III | 81.29    | 0.5192  | 0.5106 | 0.5419 |

<br>

### __[CaiT](https://arxiv.org/abs/2103.17239)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |   67.93  | 0.6726  | 0.5640 | 0.7497 |
  | Model II  |  62.97   | 0.9318  | 0.6548 | 0.7833 |
  | Model III | 69.38    | 0.5160  | 0.4986 | 0.5195 |

<br>

### __[T2TViT](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |   88.12 | 0.9706  | 0.6502 | 0.9902 |
  | Model II  |  - | -  | - | - |
  | Model III |  77.29   | 0.6515  | 0.9737 | 0.8980 |

<br>

### __[PiT](https://arxiv.org/abs/2103.16302)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  40.63   | 0.5981  | 0.5584 | 0.6580|
  | Model II  |   33.60  | 0.5351 | 0.2022 | 0.5378 |
  | Model III | 34.18    | 0.5176  | 0.5153 | 0.5294 |

<br>



<br>

# __Previous work__
### __Equivariant Convolutional Vision Transformer__ ##

An Equivariant Convolutional Vision Transformer is built for binary classification using PyTorch. Approach and strategy are discussed in [test2_e2c_vit.ipynb](./test2_e2c_vit.ipynb)

* **Dataset**: The Dataset consists of simulated strong gravitational lensing images with and without substructure. 

* **Solution**: The notebook can be open on [GoogleColab](https://colab.research.google.com/github/sachdevkartik/GSoC-2022/blob/main/test2_e2c_vit.ipynb)

* **Model Weights**: [e2cnn_vit_2022-04-04-23-41-30.pt](model/e2cnn_vit_2022-04-04-23-41-30.pt)

* **Inference**: Please use [test2_e2cnn_vit_inference.ipynb](./test2_e2cnn_vit_inference.ipynb) file for inference.

* **Results**:

    | S.No | Metric | Value |
    | --- | --- | --- |
    | 1. | Best validation accuracy | 97.10% |
    | 2. | AUC (with sub structure)  | 0.9961 |
    | 3. | AUC (without sub structure)  | 0.9975 |

<br>


### __Convolutional Vision Transformer (CvT)__ ##

An efficient Convolutional Vision Transformer (CvT) is built for binary classification using PyTorch. Approach and strategy are discussed in [test2.ipynb](./test2.ipynb)

* **Dataset**: The Dataset consists of simulated strong gravitational lensing images with and without substructure. 

* **Solution**: The notebook can be open on [GoogleColab](https://colab.research.google.com/github/sachdevkartik/GSoC-2022/blob/main/test2.ipynb)

* **Model Weights**: [ConvTransformer_2022-04-05-21-20-09.pt](model/ConvTransformer_2022-03-15-13-40-54.pt)

* **Inference**: Please use [test2_inference.ipynb](./test2_inference.ipynb) file for inference.

* **Results**:

  | S.No | Metric | Value |
  | --- | --- | --- |
  | 1. | Best validation accuracy | 98.12% |
  | 2. | AUC (with sub structure)  | 0.9988 |
  | 3. | AUC (without sub structure)  | 0.9989 |

<br>

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


