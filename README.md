# Low-Light Face Super-Resolution via Illumination, Structure, and Texture Associated Representation

Code for the paper IC-FSRDENet.

Train IC-FSRNet model:
'''python
cd IC-FSRNet
python train.py --dir_data dir_data --writer_name icfsrnet --model MYNET 
'''
Test IC-FSRNet:
'''python
cd IC-FSRNet
python test.py --dir_data dir_data --data_test test --writer_name icfsrnet-test --model MYNET 
'''

Train DENet model:
'''python
cd DENet
python train.py  -c config/denet.json
'''
Test DENet:
'''python
cd DENet
python test.py -c config/denet.json
'''
