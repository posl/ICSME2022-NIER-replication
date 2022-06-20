# ICSME2022-NIER-replication
Replication package for ICSME2022 NIER track paper (title: An Initial Analysis of Repair and Side-effect Prediction for Neural Networks).
All results of our paper can reproduce in this repository.

*require: docker*
# How to Run
## 1. Clone this repository.
```shell
# ssh
git clone git@github.com:posl/ICSME2022-NIER-replication.git
# http
git clone https://github.com/posl/ICSME2022-NIER-replication.git
```

## 2. Build a docker container and launch jupyter notebook server on the container.
```shell
./launch.sh cpu
```
- It may take some time in the first time to build a docker image.
- After this command execution, You can access jupyter notebook by typing `localhost:8888` in your browser.

## 3. The notebooks to train models are in `nnrepair/prepare`.
- You can obtain our studied trained models by running these notebooks.
- The models trained in these scripts are saved in `nnrepair/models/saved`.