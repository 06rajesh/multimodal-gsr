# Multimodal Grounded Situation Recognition using Transformers
* This is a PyTorch implementation of Multimodal Grounded Situation Recognition with Transformers.
* **MGSRTR** (**M**ultimodal **G**rounded **S**ituation **R**ecognition **T**ransformer) achieves superior performance in all evaluation metrics on the 
the multimodal datasets generated from SWiG and Flickr30k entities.
* This repository contains instructions, and code for MGSRTR. 
___
## Overview
A multimodal grounded situation recognition system receives a pair of inputs consisting of a single image
and a caption describing the activity happening on that image and outputs a verb,
a set of semantic roles, and groundings of the entities on the image. Semantic roles
of the objects in the image describe how they participate in the activity described
by the verb. Multimodal GSR will predict the verb first and depending on the verb
it will predict the nouns from a fixed set of semantic roles for each verb and finally
the groundings on the image against each noun. We presented an MGSRTR model leveraging the attention mechanism that uses the input from
this joint representation module to solve the MGSR task that we defined.

![Multimodal Grounded Situation Recognition using Transformers](./mgsrtr.png?raw=true "MGSRTR")

MGSRTR mainly consists of two components: Transformer Encoder for verb prediction, and 
Transformer Decoder for grounded noun prediction and an additional joint representation module to encode features from both modali-
ties using BERT decoder layer output to encode features from both modalities into a single vector. 

## Environment Setup
```
git clone https://github.com/06rajesh/multimodal-gsr
cd multimodal-gsr

# (Create a conda environment, activate the environment and install PyTorch via conda)
conda create --name mgsrtr python=3.9
conda activate mgsrtr

pip install -r requirements.txt

# copy the sample env file and set the desired parameters for training or inference
cp .env.sample .env
```

## Datasets
We used two different datasets with generated captions / frames to train and test our model.

### SWiG Captions
SWiG captions dataset where the captions are generated from the original SWiG dataset output 
frames. SWiG images can be downloaded from [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip).
Generated captions can be downloaded from [here](https://drive.google.com/file/d/1Nmo9ojXsjvwy7bCF0-VpcKsBY6-1QUL4/view?usp=share_link).

Extract the downloaded images and captions in a directory named `SWiG` like the following structure.

    ├── ...
    ├── SWiG                    # SWiG Dataset root directory
    │   ├── images_512          # all the SWiG images under this folder
    │   ├── combined_jsons      # all the json files extracted from the generated captions zip
    └── ...

### Flickr30k frames
Flickr30k Frames datasets where the frames were generated using the flickr30k entities captions
and frame entities were grounded using the grounding annotations. Flickr30k images can be obtained
from [here](http://hockenmaier.cs.illinois.edu/DenotationGraph/). Generated frames can be obtained
from [here](https://drive.google.com/file/d/1IGYr2XSqMeNTkDklkxpqFbPl9C9QUvii/view?usp=sharing).

Extract the downloaded images and json in a directory named `flicker30k` like the following structure.

    ├── ...
    ├── flicker30k                # Flickr30k frames Dataset root directory
    │   ├── flicker30k-images     # all the flickr30k images under this folder
    │   ├── flickr30k-jsons       # all the json files extracted from zip will be placed here
    └── ...

## Training
Our training supports setting the training parameters using a `.env` file and read the parameters
from file. Currently, it supports two `DATASET`: `swig` or `flicker30k`. This code supports training
of four different types of model (`MODEL_TYPE`). `mgsrtr`, `duel_enc_gsr`, `t5_mgsrtr`, and `gsrtr`.
Rest of the environment params are self-explanatory.

Sample `.env` file as follows:
```
DATASET=flicker30k
DEVICE=cpu
NUM_WORKERS=4
DATASET_PATH=./flicker30k
RESUME=False
START_EPOCH=0
VERSION=v6
MODEL_TYPE=mgsrtr
```

After setting the environment variables just run the following command.
```
python main.py
```

## Model Checkpoints
Model checkpoints will be provided in near future.

## Acknowledgements
Our code is modified and adapted from these amazing repositories:

* [Grounded Situation Recognition with Transformers](https://github.com/jhcho99/gsrtr)
* [Grounded Situtation Recognition](https://github.com/allenai/swig)

## Contact
Rajesh Baidya ([rajeshbaidya.006@gmail.com](mailto:rajeshbaidya.006@gmail.com?subject=[GitHub]%20Source%20Han%20Sans))

