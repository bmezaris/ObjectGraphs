# Video Event Detection with GCNs

## Requirements

* numpy
* PyTorch
* scikit-learn

## Preparation

Before training, the datasets must be preprocessed and converted to an appropriate format for efficient data loading. The dataset root directory must contain the following two subdirectories.
* ```R152_global/```: Numpy arrays of size 9x2048 containing the global frame feature vectors for each video.
* ```R152/```: Numpy arrays of size 9x50x2048 containing the appearance feature vectors of the detected frame objects for each video.

In addition, the root directory must contain the associated dataset metadata.
* The FCVID root directory must contain a ```materials/``` subdirectory with the official training/test split _FCVID\_VideoName\_TrainTestSplit.txt_ and the video event labels _FCVID\_Label.txt_.
* The YLI-MED root directory must contain the official training/test split _YLI-MED\_Corpus\_v.1.4.txt_.

## Training

To train a new model end-to-end, run
```
python train.py --dataset_root <dataset dir> [--dataset <fcvid|ylimed>]
```
By default, the model weights are saved in the ```weights/``` directory. The GCN can be used as a standalone feature extractor. To extract the GCN weights from the full model, run
```
python save_gcn.py weights/<model name>.pt model-gcn.pt  [--dataset <fcvid|ylimed>]
```

To extract the frame feature vectors using the GCN feature extractor, run
```
python extract.py model-gcn.pt --dataset_root <dataset dir> [--dataset <fcvid|ylimed>]
```
The extracted features will be saved in the ```feats/``` directory.

To train the classifier head on the extracted frame features, run
```
python train_lstm.py --feats_folder <feats dir> [--dataset <fcvid|ylimed>]
```
This script will also periodically evaluate the performance of the model.

The training parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python train.py --help``` and
```python train_lstm.py --help```.

## Evaluation

To evaluate a model, run
```
python test.py weights/<model name>.pt --dataset_root <dataset dir> [--dataset <fcvid|ylimed>]
```

## Extracted features

Features extracted during our experiments are provided in the following FTP server:
```
ftp://multimedia2.iti.gr
```
Access to the FTP server has been tested using Mozila Firefox and File Explorer ("Add a network location") in Windows 10.
To request access information please send an email to: bmezaris@iti.gr, gkalelis@iti.gr.

The data stored in the server are:
* FCVID features extracted using Faster R-CNN-based object detector to be placed in the FCVID dataset root directory (~320 GB): FCVID.z01, FCVID.z02, FCVID.z03, FCVID.z04, FCVID.z05, FCVID.z06, FCVID.z07, FCVID.z08, FCVID.z09, FCVID.zip
* YLIMED features extracted using Faster R-CNN-based object detector to be placed in the YLIMED dataset root directory (~7 GB): YLI-MED.zip
* Model trained end-to-end using the FCVID features above (~2 GB): model-fcvid.zip
* GCN standalone feature extractor trained using the FCVID features above (~70 MB): model-gcn.zip
* FCVID features extracted using the trained FCVID GCN standalone feature extractor; to be placed in the ```feats/``` directory (~12.5 MB): feats_fcvid.zip
* YLIMED features extracted using the trained FCVID GCN standalone feature extractor; to be placed in the ```feats/``` directory (~300 MB): feats_ylimed.zip
