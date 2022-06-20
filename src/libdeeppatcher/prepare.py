"""Support experiment prepairation.
"""
import keras
from typing import List, Any, Tuple
from .types import Dataset, MatrixId
from . import utils
from . import dataset


def load_model(model_name):
    """
    :param model_name
        - mnist_dense
        - mnist_convolutional
        - resnet20
    """
    model = keras.models.load_model('./models/saved/' + model_name + '.hdf5')
    return model


def load_dataset(dataset_name):
    """Load data and split.  
    - 8割は学習用。
    - 残り2割のうち、半分ずつ正常系データ、修正対象データ群として利用。
    :param dataset_name:
        - mnist_flat
        - mnist_conv
        - cifar10
    :return splited dataset
        - train_dataset
        - val_dataset
        - adv_dataset
        - test_dataset
    """
    _train_dataset, _test_dataset = dataset.load_data(dataset_name)

    _train_size = int(_train_dataset.data.shape[0])

    train_start_index = 0
    train_end_index = int(_train_size * 0.8)

    val_start_index = train_end_index
    val_end_index = val_start_index + int(_train_size * 0.1)
    
    adv_start_index = val_end_index
    adv_end_index = adv_start_index + int(_train_size * 0.1)
    

    train_dataset = Dataset(_train_dataset.data[train_start_index:train_end_index],
                            _train_dataset.label[train_start_index:train_end_index])

    val_dataset = Dataset(_train_dataset.data[val_start_index:val_end_index],
                          _train_dataset.label[val_start_index:val_end_index])

    adv_dataset = Dataset(_train_dataset.data[adv_start_index:adv_end_index],
                          _train_dataset.label[adv_start_index:adv_end_index])

    test_dataset = _test_dataset
    
    return train_dataset, val_dataset, adv_dataset, test_dataset


def prepare_repair_target_datasets(model, dataset: Dataset, n=3) -> List[Tuple[MatrixId, Dataset]]:
    """datasetから、誤認識の多い上位nクラスのデータセットを返す
    :param model:
    :param dataset:
    :return repair_target_datasets: 
    """
    confusion_rank = utils.get_mispred_confusion_rank(model, dataset)
    mispred_indices = utils.get_indices_of_mispred(model, dataset)
    repair_target_datasets: List[Tuple[MatrixId, Dataset]] = []
    for matrix_id, count in confusion_rank[:n]:
        mispred_index = mispred_indices[matrix_id.expected_label]
        mispred_dataset = utils.get_dataset_predicted_as_y(
            model,
            utils.get_dataset_by_index(dataset, mispred_index),
            matrix_id.predicted_label
        )
        repair_target_datasets.append((matrix_id, mispred_dataset))
    return repair_target_datasets


def get_positive_dataset(model, dataset: Dataset) -> Dataset:
    """datasetから正しく認識されたものだけを取り出す
    :param model:
    :param dataset:
    :return 正しく認識されたデータセット
    """
    return utils.get_positive_dataset(model, dataset)


