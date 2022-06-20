import numpy as np
from keras.models import load_model, clone_model
import os
from .types import Dataset, MatrixId
from typing import Optional, List, Dict, Tuple


NUM_OF_CLASS = 10


def _get_index_of_cls(label: Dataset.label, cls: int) -> List[int]:
    """Get index of specified cls.
    :param label(ndarray): label
    :param cls: 0 ~ 9
    :return index: index of specified cls.
    """
    return (np.where(label[:, cls] == 1)[0]).tolist()


def _get_index_of_cls_mispred(model, dataset: Dataset, digit: int) -> List[int]:
    """Get index of mispredicted data from specified digit.
    :param model (keras model):
    :param dataset: data and label
    :param digit: 0 ~ 9
    :return index_mispred: index of mispredicted data from specified digit.
    """
    index_digit = _get_index_of_cls(dataset.label, digit)
    index_mispred = []
    for i in index_digit:
        predicted_value = model.predict(dataset.data[i:i+1])[0].argmax()
        if np.where(dataset.label[i] == 1) != np.array(predicted_value):
            index_mispred.append(i)
    return index_mispred


def get_indices_of_mispred(model, dataset: Dataset) -> List[List[int]]:
    """Get mispredicted data index of each label.
    :param model (keras model):
    :param dataset: data and label
    :return indices_mispred:
        mispredicted data indices.
        e.g. indices_mispred[i] => index of mispredicted class i
    """
    indices_mispred = []
    for i in range(NUM_OF_CLASS):
        index = _get_index_of_cls_mispred(model, dataset, i)
        indices_mispred.append(index)
    return indices_mispred


def get_dataset_by_index(dataset: Dataset, index: List[int]):
    """Get dataset selected by index.
    :param dataset: data and label
    :param index: dataset index
    :return and 
    """
    data = dataset.data[index]
    label = dataset.label[index]
    return Dataset(data, label)


def get_dataset_by_cls(dataset: Dataset, cls: int) -> Dataset:
    """Get dataset selected by dataset class.
    :param dataset: data and label.
    :param cls: class of dataset
    :return dataset of specified class
    """
    index = _get_index_of_cls(dataset.label, cls)
    return get_dataset_by_index(dataset, index)


def get_dataset_by_random_sampling(dataset: Dataset, num=None) -> Dataset:
    """
    """
    dsize = len(dataset.data)
    _sample_indices = np.random.choice(dsize, num).tolist()
    return get_dataset_by_index(dataset, _sample_indices)


def get_positive_dataset(model, dataset: Dataset) -> Dataset:
    """Get dataset predicted correctly.
    :param model:
    :param dataset:
    :return dataset predicted correctly
    """
    preds = model.predict(dataset.data).argmax(axis=1)
    pos_index = np.where(preds == dataset.label.argmax(axis=1))
    positive_dataset = get_dataset_by_index(dataset, pos_index)
    return positive_dataset


def get_negative_dataset(model, dataset: Dataset) -> Dataset:
    pass


def get_dataset_predicted_as_y(model, dataset: Dataset, y) -> Dataset:
    """Get dataset that model predicts as y.
    :param model:
    :param dataset:
    :param y: predicted label
    :return dataset that model predicts as y.
    """
    preds = model.predict(dataset.data).argmax(axis=1)
    index_predicted_as_y = np.where(preds == y)[0].tolist()
    dataset_predicted_as_y = get_dataset_by_index(dataset, index_predicted_as_y)
    return dataset_predicted_as_y


def get_result_score(model, dataset: Dataset) -> Dict[str, np.float64]:
    """Get scores of model predictions to dataset.
    :param model: model to evaluate
    :param dataset: data and label to predict
    :return scores (dict): { 'label', 'score' }
    """
    scores = {}
    total_score = model.evaluate(dataset.data, dataset.label, verbose=0)[1]
    scores['total'] = total_score
    for i in range(NUM_OF_CLASS):
        cls_dataset = get_dataset_by_cls(dataset, i)
        score = model.evaluate(cls_dataset.data,
                               cls_dataset.label,
                               verbose=0)[1]
        scores[str(i)] = score
    return scores


def get_mispred_confusion_rank(model, dataset: Dataset) -> List[Tuple[MatrixId, int]]:
    """confusion_matrixを、誤認識が多い要素から順にソートし、リストとして返す。
    expected_label: datasetに与えられたラベル（想定解）
    actual_pred: 実際に予測されたラベル
    :param model:
    :param dataset:
    :return
    """
    mispred_matrix = {MatrixId(i, j): 0
                      for i in range(10)
                      for j in range(10)}
    mispred_indices = get_indices_of_mispred(model, dataset)
    for expected_label, mispred_index in enumerate(mispred_indices):
        if mispred_index == []:
            continue
        mispred_dataset = get_dataset_by_index(dataset, mispred_index)
        for predicted_label in model.predict(mispred_dataset.data).argmax(axis=1):
            mispred_matrix[MatrixId(expected_label, predicted_label)] += 1

    mispred_matrix_list = list(mispred_matrix.items())
    mispred_matrix_list.sort(key=lambda x: x[1], reverse=True)
    return mispred_matrix_list


def get_trainable_layer_indexs(model) -> List[int]:
    """Get trainable layer indexs.
    :param model: kears model
    :return trainable_layer_indexs: index list of trainable layers of model
    """
    trainable_layer_indexs = []
    for i, layer in enumerate(model.layers):
        layer_config = layer.get_config()
        if layer_config['trainable']:
            trainable_layer_indexs.append(i)
    return trainable_layer_indexs


def get_repairable_layer_indexs(model) -> List[int]:
    """Get repairable layer indexs.
    :param model: keras model
    :return repairable_layer_indexs: index list of repairable layers of model
    """
    repairable_layer_indexs = []
    for i, layer in enumerate(model.layers): 
        config_keys = layer.get_config().keys()
        # dense
        if 'units' in config_keys:
            repairable_layer_indexs.append(i)
        # conv
        if 'filters' in config_keys:
            repairable_layer_indexs.append(i)
    return repairable_layer_indexs


def get_repairable_dense_layer_indices(model) -> List[int]:
    """Get dense layer indices which have parameters.
    :param model:
    :return layer indices
    """
    repairable_dense_layer_indices = []
    for i, layer in enumerate(model.layers):
        config_keys = layer.get_config().keys()
        # the key of 'units' means dimension of output
        if 'units' in config_keys:
            repairable_dense_layer_indices.append(i)
    return repairable_dense_layer_indices


def copy_model(model):
    """return model copy
    :param model: copy target model
    :return model_copy: copy of model
    """
    tmp = './tmp_model.hdf5'
    model.save(tmp)
    model_copy = load_model(tmp)
    os.remove(tmp)
    return model_copy


def copy_model_ver2(model):
    cln_model = clone_model(model)
    cln_model.build((None, 784))
    cln_model.compile(optimizer=model.optimizer,
                      loss=model.loss,
                      metrics=model.metrics)
    cln_model.set_weights(model.get_weights())
    return cln_model


def build_model(model):
    _model = clone_model(model)
    _model.build(model.input.shape)
    _model.compile(optimizer=model.optimizer,
                   loss=model.loss,
                   metrics=model.metrics)
    return _model
