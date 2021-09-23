# McGANn

McGAN is a conditional Generative Adversarial Network for generating Gravitational-Wave burst signals. You can read the associated paper [here](https://iopscience.iop.org/article/10.1088/1361-6382/ac09cc).

## Requirements

Dependency's are found in [requirements.txt](https://github.com/jmcginn/McGANn/blob/master/cGAN/requirements.txt)

Current version working on Python 3.9.7 and Tensorflow 2.6.0. Use of a GPU is recommended.

## Usage
Git clone the repo to your desired directory. Change the following paths in [McGANn.py](https://github.com/jmcginn/McGANn/blob/master/cGAN/McGAN.py) to where you would like the plots saved:
```
# directory for the loss plot
loss_dir = ''
# directory for the generations
gen_dir = ''
# directory for the training data examples
train_dir = ''
# directory for saving the trained model
gen_model_dir = ''
```

To train:

```
python McGANn.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
