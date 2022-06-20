# ICSME2022-NIER-replication
Replication package for ICSME2022 NIER track paper (title: An Initial Analysis of Repair and Side-effect Prediction for Neural Networks).
All results of our paper can reproduce in this repository.

*require: docker, make*
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

## 3. Get our trained models.
- You can obtain our studied trained models by running the notebooks in `src/prepare`.
- These trained models are saved in `src/models/saved`, so you can obtain these models without executing above scripts.

## 4. Apply Arachne for each model and fault.
```shell
pwd
# /XXX/ICSME2022-NIER-replication
bash arachne_apply.sh
```
- It may take long time...
- model files that are applied arachne is saved under `src/arachne_results`.

## 5. Get the repairs / side-effect dataset


## 6. Build the prediction models for repairs / side-effect.
