# Video Event Detection with GCNs

This repository hosts the code and data for our paper: N. Gkalelis, A. Goulas, D. Galanopoulos, V. Mezaris, "ObjectGraphs: Using Objects and a Graph Convolutional Network for the Bottom-up Recognition and Explanation of Events in Video", Proc. 2nd Int. Workshop on Large Scale Holistic Video Understanding (HVU) at the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), June 2021.

## Code requirements

* numpy
* PyTorch
* scikit-learn

## Video preprocessing

Before training our method on any video dataset, the videos must be preprocessed and converted to an appropriate format for efficient data loading (In our work, we sample xxx frames per video; on each frame, a variant of the Faster R-CNN object detector is used [3,4] for object detection, and an xxx network is used for extracting a representation of each entire frame as well as each object region). Following video preprocessing, the dataset root directory must contain the following two subdirectories:
* ```R152_global/```: Numpy arrays of size 9x2048 containing the global frame feature vectors for each video (the 9 frames, times the 2048-element vector for each frame).
* ```R152/```: Numpy arrays of size 9x50x2048 containing the appearance feature vectors of the detected frame objects for each video (the 9 frames, times the 50 most-promiment objects identified by the obejct detector, times a 2048-element vector for each object bounding box).

In addition, the root directory must contain the associated dataset metadata:
* The FCVID root directory must contain a ```materials/``` subdirectory with the official training/test split _FCVID\_VideoName\_TrainTestSplit.txt_ and the video event labels _FCVID\_Label.txt_.
* The YLI-MED root directory must contain the official training/test split _YLI-MED\_Corpus\_v.1.4.txt_.

## Training

To train a new model end-to-end, run
```
python train.py --dataset_root <dataset dir> [--dataset <fcvid|ylimed>]
```
By default, the model weights are saved in the ```weights/``` directory. The trained GCN can also be used as a standalone feature extractor. To extract the GCN weights from the full model, run
```
python save_gcn.py weights/<model name>.pt model-gcn.pt  [--dataset <fcvid|ylimed>]
```

To extract the frame feature vectors using the GCN feature extractor, run
```
python extract.py model-gcn.pt --dataset_root <dataset dir> [--dataset <fcvid|ylimed>]
```
The extracted features will be saved in the ```feats/``` directory.

To train the classifier head on the GCN-extracted frame features, run
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

## Usage

To run the code for the different datasets (FCVID, YLI-MED) use the corresponding settings described in the paper.
For instance, to train the model end-to-end and evaluate it using the FCVID dataset, run
```
python train.py --dataset_root <FCVID root directory> --dataset fcvid --num_epochs 60 --step_size 50 --lr 1e-4 --batch_size 64
```
```
python test.py weights/model-fcvid-060.pt --dataset_root <FCVID root directory> --dataset fcvid
```
To extract the GCN standalone feature extractor from the FCVID trained model, use the GCN standalone feature extractor to extract frame-level YLI-MED features and train the LSTM classifier head, run
```
python save_gcn.py weights/model-fcvid-060.pt model-gcn.pt  --dataset fcvid
```
```
python extract.py model-gcn.pt --dataset_root <YLI-MED root directory> --dataset ylimed
```
```
python train_lstm.py --feats_folder feats --dataset ylimed --num_epochs 30 --lr 1e-4 --batch_size 16 --gamma 0.9
```

## Provided features

Features extracted during our experiments are provided in the following FTP server:
```
ftp://multimedia2.iti.gr
```
Access to the FTP server has been tested using Mozila Firefox and File Explorer ("Add a network location") in Windows 10.
To request access creadentials for the ftp please send an email to: bmezaris@iti.gr, gkalelis@iti.gr.

The data stored in the ftp server are:
* FCVID features extracted using Faster R-CNN-based object detector to be placed in the FCVID dataset root directory (~320 GB): FCVID.z01, FCVID.z02, FCVID.z03, FCVID.z04, FCVID.z05, FCVID.z06, FCVID.z07, FCVID.z08, FCVID.z09, FCVID.zip
* YLIMED features extracted using Faster R-CNN-based object detector to be placed in the YLIMED dataset root directory (~7 GB): YLI-MED.zip
* Model trained end-to-end using the FCVID features above (~2 GB): model-fcvid.zip
* GCN standalone feature extractor trained using the FCVID features above (~70 MB): model-gcn.zip
* FCVID frame features extracted using the trained FCVID GCN standalone feature extractor; to be placed in the ```feats/``` directory (~12.5 GB): feats_fcvid.zip
* YLI-MED frame features extracted using the trained FCVID GCN standalone feature extractor; to be placed in the ```feats/``` directory (~300 MB): feats_ylimed.zip

## License and Citation

The code of our ObjectGraphs method is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources (e.g. provided datasets [1,2], etc.). If you find the ObjectGraphs code useful in your work, please cite the following publication where this approach was proposed:

N. Gkalelis, A. Goulas, D. Galanopoulos, V. Mezaris, "ObjectGraphs: Using Objects and a Graph Convolutional Network for the Bottom-up Recognition and Explanation of Events in Video", Proc. 2nd Int. Workshop on Large Scale Holistic Video Understanding (HVU) at the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), June 2021.

Bibtex:
```
@INPROCEEDINGS{FSDP_ICMEW2020,
               AUTHOR    = "N. Gkalelis and A. Goulas and D. Galanopoulos and V. Mezaris",
               TITLE     = "ObjectGraphs: Using Objects and a Graph Convolutional Network for the Bottom-up Recognition and Explanation of Events in Video",
               BOOKTITLE = "Proc. IEEE CVPR Workshops (CVPRW)",
               ADDRESS   = "",
               PAGES     = "",
               MONTH     = "June",
               YEAR      = "2021"
}
```

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreements 832921 (MIRROR) and 951911 (AI4Media).

## References

[1] YY.-G. Jiang, Z. Wu et al. Exploiting feature and class relationships in video categorization with regularized deep neural networks. IEEE Trans. Pattern Anal. Mach. Intell., 40(2):352–364, 2018

[2] J. Bernd, D. Borth et al. The YLI-MED corpus: Characteristics, procedures, and plans. CoRR, abs/1503.04250, 2015.

[3] P. Anderson, X. He et al. Bottom-up and top-down attention for image captioning and visual question answering. In Proc. ICVGIP, pages 6077–6086, Hyderabad, India, Dec. 2018

[4] S. Ren, K. He et al. Faster R-CNN: Towards real-time object detection with region proposal networks. In Proc. NIPS, volume 28, 2015.
