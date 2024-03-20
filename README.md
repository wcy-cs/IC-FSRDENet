# Low-Light Face Super-resolution via Illumination, Structure, and Texture Associated Representation

Code for the paper IC-FSRDENet.

## Requirement
Pytorch 1.11.0 Cuda 11.4 

Train IC-FSRNet model:

```Python
cd IC-FSRNet
python train.py --dir_data dir_data --writer_name icfsrnet --model MYNET 
```
Test IC-FSRNet:
```Python
cd IC-FSRNet
python test.py --dir_data dir_data --data_test test --writer_name icfsrnet-test --model MYNET 
```

Train DENet model:
```Python
cd DENet
python train.py  -c config/denet.json
```
Test DENet:
```Python
cd DENet
python test.py -c config/denet_test.json
```
## Test Dataset
[BaiDu](https://pan.baidu.com/s/1D1LOou8CKrjy3v-vXPp59A) passward:x6y7 

## Pretrained Model
[BaiDu](https://pan.baidu.com/s/1Oe95lTX6xQ4NzaSG1ZOIeQ) passward:ywqj 


## Citation 
```
@InProceedings{Wang_2024_AAAI,
    author    = {Wang, Chenyang and Jiang, Junjun and Jiang, Kui and Liu, Xianming},
    title     = {Spatial-Frequency Mutual Learning for Face Super-Resolution},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    year      = {2024},
}
```
