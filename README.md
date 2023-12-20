# Low-Light Face Super-Resolution via Illumination, Structure, and Texture Associated Representation

Code for the paper IC-FSRDENet.

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
## Dataset
[BaiDu](https://pan.baidu.com/s/1LX66EKkx51G7ZUAL4MYgRw) passward:nasx 
## Pretrained Model
[BaiDu](https://pan.baidu.com/s/1Oe95lTX6xQ4NzaSG1ZOIeQ) passward:ywqj 
