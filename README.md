# Paraphrase Detection with BiMPM

## Description
This repository contains the [PyTorch](http://pytorch.org/) implementation of the Bilateral Multi-perspective Matching model [BiMPM](https://arxiv.org/pdf/1702.03814.pdf)) described in the paper by Wang et al. The model is used to perform a paraphrase detection task on the [Quora Questions Pairs dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs). In order to maintain consistency in comparison, we adopted the train/dev/test partition by Wang et al. The program takes two phrases as inputs and predicts a value to indicate if the two phrases are paraphrases of each other or not.
<img src="https://github.com/timatim/ParaphraseDetection/blob/master/BiMPM.png" data-canonical-src="https://github.com/timatim/ParaphraseDetection/blob/master/BiMPM.png" width="500" height="400" />

## Requirements
 - python 3.5
 - torch 0.1.12

## Train
To train the model using the setting described in the paper, run
> python trainer.py --embedding wordvec.txt --data quora_data/ --word-len 15 --seq-len 50 --perspectives 5 --batch-size 32 --cuda 

## Test
To test a model on the test dataset, run
> python test.py --embedding wordvec.txt --data quora_data/test.tsv --word-len 15 --seq-len 50 --perspectives 5 --batch-size 32 --model model.pth

## Issues
Please report any issues to me juiting.hsu@nyu.edu.

## Reference
 - Zhiguo Wang, Wael Hamza, Radu Florian. [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf). IJCAI (2017)
 - [Quora Question Pairs Dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
