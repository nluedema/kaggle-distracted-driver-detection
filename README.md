# Kaggle Competition - State Farm Distracted Driver Detection
This repository contains code to train models for the State Farm Distracted Driver Detection competition on Kaggle. For more information on the competition see [here](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

## Results
Below you can find the results that I got when training the models. These results are not part of the competition's leaderboard, as they were submitted after the end of the competition. Therefore, the placements are just hypothetical.

| Model | Private <br/> Score | Private <br/> Placement | Public <br/> Score | Public <br/> Placement |
| --- | --- | --- | --- | --- |
| ResNet34 | 0.34788 | 233/1438 | 0.37174 | 248/1438 |
| ResNet50 | 0.28872 | 182/1438 | 0.32673 | 214/1438 |

## Usage
```sh
git clone https://github.com/nluedema/kaggle-distracted-driver-detection.git
cd kaggle-distracted-driver-detection
pip install -r requirements.txt

# download dataset and perform train/val split
cd data
sh download_data_and_split.sh <KAGGLE_USERNAME> <KAGGLE_KEY>
cd ..

# train models
python code/run.py
```
Notice that `download_data_and_split.sh` expects the `unzip` cli tool to be installed.

## Output
The results of `run.py` are stored in the folder `experiments`. For each model a folder of the form `<timestamp_modelname>` is created. This folder contains a log, the accuracy values, the loss values and the best model for the training and the fine-tuning run. As the fine-tuning does not use a validation set, the best model corresponds to the one of the last epoch. Additionally, a submission file with predictions for the test set is created using the fine-tuned model.

## Models
Two models are trained, one is based on a pretrained ResNet34 the other is based on a pretrained ResNet50. For both models the same procedure is used. First, the last two layers (adaptive avg pool and linear) are removed. Then a layer that returns the concatenation of adaptive avg pool and adaptive max pool is added. For more information on this layer see [here](https://forums.fast.ai/t/what-is-the-distinct-usage-of-the-adaptiveconcatpool2d-layer/7600). Finally, two blocks of [BatchNorm,Dropout,Linear] are added. The first linear layer uses ReLU as its activation function, while the second does not use any activation function, as its output is directly passed to the loss. The implementation of the models can be found in `code/model.py`.

## Training
The training is performed in two steps. First, only the last conv block of the ResNet and the newly added components are trained. The purpose of this step is to learn a classifier that can use the image features of the pretrained ResNet to make good predictions for the task at hand. This step uses the created train/val split.

In the second step, the model that performed best in terms of validation loss is fine-tuned. In this step the whole model is tuned and the combined train and validation set is used for training. The purpose of this step is to utilize all of the available data for the final model and to allow the ResNet to adapt its image feature generation to the task at hand. It should be noted that this step is prone to overfitting, as no validaton set is used as a control measure.





