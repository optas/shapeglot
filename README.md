# ShapeGlot: Learning Language for Shape Differentiation
Created by <a href="https://ai.stanford.edu/~optas" target="_blank">Panos Achlioptas</a>, <a href="https://cogtoolslab.github.io/people.html" target="_blank">Judy Fan</a>, <a href="https://rxdhawkins.com" target="_blank">Robert X.D. Hawkins</a>, <a href="https://cocolab.stanford.edu/ndg.html" target="_blank">Noah D. Goodman</a>, <a href="https://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a>.

![representative](https://github.com/optas/shapeglot/blob/master/doc/images/teaser.jpg)


## Introduction
This work is based on our ICCV-2019 [paper](https://arxiv.org/abs/1905.02925). 
There, we proposed <i>speaker \& listener neural models</i> that reason and differentiate objects according to their <i>shape</i> via language (hence the term shape<i>--glot</i>).
These models can operate on <b>2D images and/or 3D point-clouds</b> and do learn about natural properties of shapes, including the part-based 
compositionality of 3D objects, from language <b>alone</b>. The latter fact, makes them remarkably robust, enabling a plethora of <i>zero-shot-transfer</i> learning applications. You can check our [project's webpage](https://ai.stanford.edu/~optas/shapeglot) for a quick introduction and produced results.


## Dependencies
Main Requirements:
- Python 3x (with numpy, pandas, matplotlib, nltk) 
- [Pytorch (version 1.0+)](https://pytorch.org)

Our code has been tested with Python 3.6.9, Pytorch 1.3.1, CUDA 10.0 on Ubuntu 14.04.

## Installation
Clone the source code of this repository and pip install it inside your (virtual) environment. 
```
git clone https://github.com/optas/shapeglot
cd shapeglot
pip install -e .
```

### Data Set
We provide 78,782 utterances referring to a ShapeNet chair that was contrasted against two distractor chairs via the 
reference game described in our accompanying paper (dataset termed as ChairsInContext). We further provide the data used in the Zero-Shot experiments which include
300 images of real-world chairs, and 1200 referential utterances for ShapeNet lamps & tables & sofas, and 400 utterances describing ModelNet beds.
Last, we include image-based (VGG-16) and point-cloud-based (PC-AE) pretrained features for all ShapeNet chairs to facilitate the training of the neural speakers and listeners.  
      
To download the data (~218 MB) please run the following commands. Notice, that you first need to accept the Terms Of Use [here](https://docs.google.com/forms/d/e/1FAIpQLScyV1AsZsfthqiPhuw6MFL1JZ4p8GSDPIj8uwH0BRWQl3tejw/viewform). Upon review we will email to you the necessary link that you need to put inside the desingated location of the download_data.sh file.
```
cd shapeglot/
./download_data.sh
```
The downloaded data will be stored in shapeglot/shapeglot_data


### Usage
To easily expose the main functionalities of our paper, we prepared some simple, instructional notebooks.

1. To tokenize, prepare and visualize the chairsInContext dataset, please look/run:
```
    shapeglot/notebooks/prepare_chairs_in_context_data.ipynb
```

2. To train a neural listener (**only ~10 minutes** on a single modern GPU):
```
    shapeglot/notebooks/train_listener.ipynb
```

**Note:** This repo contains limited functionality compared to what was presented in the paper. This is because our original 
(much heavier) implementation is in low-level TensorFlow and python 2.7. If you need more functionality (e.g. pragmatic-speakers) 
and you are OK with Tensorflow, please email panos@cs.stanford.edu .

## Citation
If you find our work useful in your research, please consider citing:

	@article{shapeglot,
	  title={ShapeGlot: Learning Language for Shape Differentiation},
	  author={Achlioptas, Panos and Fan, Judy and Hawkins, Robert X. D. and Goodman, Noah D. and Guibas, Leonidas J.},
	  journal={CoRR},
	  volume={abs/1905.02925},
	  year={2019}
	}


## License
This provided code is licensed under the terms of the MIT license (see LICENSE for details).
