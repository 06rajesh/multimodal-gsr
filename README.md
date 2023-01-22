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