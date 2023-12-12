This repository contains code to train a video classifier based on extracted I3D features.

To train this model, update the path to the dataset, labels, and the logs directory in `run.sh` and run the script:
```
./run.sh
```

Once you have trained the model, you can evaluate on the test set using the following command after updating the path to the trained checkpoint and the test set and labels (if available):

```
./test.sh
```

The code was tested with the following versions:
```
torch==1.9.0
pytorch-lightning==1.2.10
```
