# arae

A clean implementation of "Adversarially Regularized Autoencoders (ICML 2018)" by Zhao, Kim, Zhang, Rush and LeCun https://arxiv.org/abs/1706.04223

To evaluate a model you can download a pretrained kenlm model (the model trained on the same train.txt file):

* SNLI model: https://drive.google.com/file/d/1zpVYD9USw8fxrTl1VzSTKpBHgUaUr5XW/view?usp=sharing


### SNLI training:

```console
python train.py --data data_snli --no_earlystopping --gpu 0 --kenlm_model knlm_snli.arpa
```

![snli_training.png](https://github.com/awant/arae/blob/master/imgs/snli_train.png?raw=true)

![snli_testing.png](https://github.com/awant/arae/blob/master/imgs/snli_test.png?raw=true)

1. After 1 epoch:
  * the people are looking at food .
  * the little boy is going to catch the grass .
  * the old man is not being .
  * the man and woman are having in a kitchen .
  * four girls walk along a street while others watch
2. After 2 epochs:
  * two women are in a crowd .
  * couple sitting with their women working .
  * a man can be walking through the grass in the mountains .
  * a man is trying to buy down a wall
  * women in a large lab room
3. After 5 epochs:
  * a group of adults with a clean up .
  * a bike is decorated in a mountain station .
  * people are playing rugby .
  * a man is in his hand of a t-shirt .
  * a man tries to prepare for the lake .
4. After 10 epochs:
  * a basketball player dancing on the beach
  * a man at a tall go getting .
  * a old woman his scooter before racing .
  * the female is wearing red bicycle for a snowmobile .
  * a little girl in a red scarf is bed and sleeping on his room .


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

