# ReadMe

This is the source code for our IEEE Access publication *MILE: Memory-Interactive Learning Engine for Neuro-Symbolic Solutions to Mathematical Problems*. 

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

To switch to other encoder and decoder models, modify the `encode_method` and `decode_method` in `config.py`, or assign `@encode_method="..." @decode_method="..."` in the training command:

```
python ./train.py @encode_method="..." @decode_method="..."
```
