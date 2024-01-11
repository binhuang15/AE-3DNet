# AE-3DNet
This is the offical implementation of AE-3DNet in our work entitled 

"Prediction of the response to a combination of HAIC with systematic therapy for HCC via deep learning from pretreatment CEUS cine: A prospective dual-center study". 

File descriptions are as follows:
* [kfold_train.py](https://github.com/mp31192/AE-3DNet/blob/main/kfold_train.py): This code is for model training.
* [kfold_test.py](https://github.com/mp31192/AE-3DNet/blob/main/kfold_test.py): This code is for model testing.
* [model.py](https://github.com/mp31192/AE-3DNet/blob/main/model.py): This code corresponds to our implementation of AE-3DNet.

How to start
=============
This code can be easily performed on CEUS data with classification anotation. Here, we split the whole process into 4 steps so that you can reproduce the AE-3DNet on your personal side.

* Step 1: Environment setting.
* Step 2: Data preparation.
* Step 3: Model training. Run the script [kfold_train.py](https://github.com/mp31192/AE-3DNet/blob/main/kfold_train.py) to train the AE-3DNet on training set and validation set.
* Step 4: Model evaludation. Run the script [kfold_test.py](https://github.com/mp31192/AE-3DNet/blob/main/kfold_test.py) to test the performance of AE-3DNet on test set.

Step 1: Environment setting
-------------
Our code was implemented in following environment：

* Python == 3.7.0
* torch == 1.7.0
* cuda == 11.0
* GPU == Nvidia GTX 1080Ti with 11GB memory

Step 2: Data preparation
-------------
Prepare the data according to our paper, make sure that the data for each patient comprise 81 CEUS image crops.
You need to split the patients into five folds to yield a training set, a validation set, and a test set.
Then, save the data path and the corresponding label in a `.npy` file for each set. For example:
* train.npy  ---  [["train_path_1", "0"], ["train_path_2", "1"], ..., ["train_path_N", "0"]]
* val.npy    ---  [["val_path_1", "1"],   ["val_path_2", "0"],   ..., ["val_path_N", "0"]]
* test.npy   ---  [["test_path_1", "1"],  ["test_path_2", "0"],  ..., ["test_path_N", "1"]]

These three `.npy` files should be placed in a folder and the files of five folds should be save in a folder `ALL_Data` as follows:
<table>
  <tr>
    <td>0</td>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
  </tr>
    <tr>
    <td>train.npy val.npy test.npy</td>
    <td>train.npy val.npy test.npy</td>
    <td>train.npy val.npy test.npy</td>
    <td>train.npy val.npy test.npy</td>
    <td>train.npy val.npy test.npy</td>
  </tr>
</table>
  
Step 3: Model training
-------------
Run the script [kfold_train.py](https://github.com/mp31192/AE-3DNet/blob/main/kfold_train.py) to train AE-3DNet using `train.npy` and `val.npy` files. The `train.npy` contains training set for model training and the `val.npy` contains validation set for model selection.

You can run the script as follows:
```Python
python kfold_train.py --data_path ./ALL_Data
```

During model training, the checkpoints of the best performance on the validation set are saved in the folder `model_checkpoint`.

Step 4: Model evaluation
-------------
Run the script [kfold_test.py](https://github.com/mp31192/AE-3DNet/blob/main/kfold_test.py) to test the performance of AE-3DNet using `test.npy`. The `test.npy` contains test set for model testing.

Before runing the script, you need to select the checkpoint with the best performance from each fold and move them into a folder `model_for_test`.
You can run the script as follows:
```Python
python kfold_test.py --data_path ./ALL_Data --model_save_name ./model_for_test
```

Finally, you can obtain the ROC and corresponding AUC of AE-3DNet on the test set.
