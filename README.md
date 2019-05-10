# Arae

A clean implementation of "Adversarially Regularized Autoencoders (ICML 2018)" by Zhao, Kim, Zhang, Rush and LeCun https://arxiv.org/abs/1706.04223

To evaluate a model you can download a pretrained kenlm model (the model trained on the same train.txt files):

* SNLI model: https://drive.google.com/file/d/1zpVYD9USw8fxrTl1VzSTKpBHgUaUr5XW/view?usp=sharing


### SNLI training:

```console
python train.py --data data_snli --no_earlystopping --gpu 0 --kenlm_model knlm_snli.arpa
```

##### Additional options:

| option             | description                                             |
|--------------------|---------------------------------------------------------|
| --tensorboard      | draw graphs. need tensorboardx to work                  |
| --kenlm_model      | path to reference kenlm model for computing forward ppl |
| --gpu              |  -1 - don't use gpu, > -1 - use                         |
| --compressing_rate | -S param for kenlm cmd line util                        |


### Generating sentences:

```console
python generate.py --greedy
```

