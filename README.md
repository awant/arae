# Arae

A clean implementation of "Adversarially Regularized Autoencoders (ICML 2018)" by Zhao, Kim, Zhang, Rush and LeCun https://arxiv.org/abs/1706.04223

To evaluate a model you can download a pretrained kenlm model (the model trained on the same train.txt files):

* SNLI model: https://drive.google.com/file/d/1zpVYD9USw8fxrTl1VzSTKpBHgUaUr5XW/view?usp=sharing


### SNLI training:

```console
python train.py --data data_snli --no_earlystopping --gpu 0 --kenlm_model knlm_snli.arpa
```

![snli_training.png](https://github.com/awant/arae/blob/master/imgs/snli_train.png?raw=true)

![snli_testing.png](https://github.com/awant/arae/blob/master/imgs/snli_test.png?raw=true)

1. After 1 epoch:
  1.1 the people are looking at food .
  1.2 the little boy is going to catch the grass .
  1.3 the old man is not being .
  1.4 the man and woman are having in a kitchen .
  1.5 four girls walk along a street while others watch
2. After 2 epochs:
  2.1 two women are in a crowd .
  2.2 couple sitting with their women working .
  2.3 a man can be walking through the grass in the mountains .
  2.4 a man is trying to buy down a wall
  2.5 women in a large lab room
3. After 5 epochs:
  3.1 a group of adults with a clean up .
  3.2 a bike is decorated in a mountain station .
  3.3 people are playing rugby .
  3.4 a man is in his hand of a t-shirt .
  3.5 a man tries to prepare for the lake .
4. After 10 epochs:
  4.1 a basketball player dancing on the beach
  4.2 a man at a tall go getting .
  4.3 a old woman his scooter before racing .
  4.4 the female is wearing red bicycle for a snowmobile .
  4.5 a little girl in a red scarf is bed and sleeping on his room .


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

