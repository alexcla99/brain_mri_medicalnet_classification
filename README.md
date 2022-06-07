# Fine-tuning a medical-wise CNN for brain MRIs classification.

Author: alexcla99
Version: 1.0.0

### Folder content:

```
+--- data/                              # The folder containing brain MRIs
|
+-+- models/                            # The folder containing models
| +--- resnet.py                        # The base model builder
| +--- resnet_10.pth                    # The pretrained Resnet model with 10 layers
| +--- resnet_18.pth                    # The pretrained Resnet model with 18 layers
| +--- resnet_34.pth                    # The pretrained Resnet model with 34 layers
| +--- resnet_50.pth                    # The pretrained Resnet model with 50 layers
| +--- resnet_101.pth                   # The pretrained Resnet model with 101 layers
| +--- resnet_152.pth                   # The pretrained Resnet model with 152 layers
| +--- resnet_200.pth                   # The pretrained Resnet model with 200 layers
|
+--- results/                           # The folder containing both train and test results
+--- __init__.py                        # An empty file to make this directory being a Python library
+--- dataset.py                         # The dataset loader
+--- model.py                           # The fine-tuned model builder
+--- preprocess_to_numpy.py             # A script to preprocess the dataset and store it into numpy files
+--- README.md                          # This file
+--- requirements.txt                   # The Python libraries to be installed in order to run the project
+--- settings.json                      # The settings of the model and the train phase
+--- test.py                            # A script to test the fine-tuned model performances
+--- train.py                           # A script to train the fine-tuned model
+--- utils.py                           # Some utils
```

### Usage:

This library has been implemented and used with Python>=3.8.0

Requirements:
```Shell
pip3 install -r requirements
```

Preprocess data:
```Shell
python3 preprocess_to_numpy.py
```
Data are loaded from both "coma" and "control" subdirectories (from "data") in order to get stored in numpy files.

Fine-tune the model:
```Shell
python3 train.py python3 train.py resnet_<layers:int> <debug:bool>
# Example: python3 train.py resnet_50 False
```
Data to be used are selected from the "data" folder and results are saved in the "results" folder.

Available networks:
See the `models` folder.

<u>Note</u>: both resnet_18 and resnet_34 pretrained models have not been tested in this project because of missing values in their state dictionnary (for classification task). Resnet_10, resnet_101, resnet_152 and resnet_200 have been fine-tuned and tested.

Test the model:
```Shell
python3 test.py resnet_<layers:int> <debug:bool>
# Example: python3 test.py resnet_101 False
```
<u>Note</u>: the layers specified should match the subdirectory `results/r<layers>` such as "results/r101" if you selected the 'resnet_101' model.
Data to be used are selected from the "data" folder and results are saved in the "results" folder.

### Many thanks to:

```Bib
@article{
    chen2019med3d,
    title={Med3D: Transfer Learning for 3D Medical Image Analysis},
    author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
    journal={arXiv preprint arXiv:1904.00625},
    year={2019}
}
```

License: MIT.