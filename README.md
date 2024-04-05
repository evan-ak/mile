# ReadMe

This is the source code for our paper under submission *MILE: Memory-Interactive Learning Engine for Neuro-Symbolic Solutions to Mathematical Problems*. 

<br/>

# Environment

The code was written and verified on Python 3.11 and PyTorch 2.1.0. 

The other required libraries include :

+ json
+ numpy
+ transformers

<br/>

# Verification

To start the training, run:

```
python ./train.py
```

To switch to other encoder and decoder model, modify the `encode_method` and `decode_method` in `config.py`, or assign `@encoder_method="..." @decoder_method="..."` in the training command:

```
python ./train.py @encoder_method="..." @decoder_method="..."
```