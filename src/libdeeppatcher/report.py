import numpy as np
from . import utils
from . import plot
from typing import List
from .types import Dataset


NUM_OF_CLASS = 10


def _get_actual_labels(model, data: Dataset.data) -> np.ndarray:
    """Get actual predcted labels.
    """
    return np.argmax(model.predict(data), axis=1)


def _get_expected_labels(label: Dataset.label) -> np.ndarray:
    """Get expected labels.
    """
    return np.argmax(label, axis=1)


def _get_index_of_neq(predA: np.ndarray, predB: np.ndarray) -> List[int]:
    """predAとpredBで結果の異なるインデックスを返す
    :param predA, predB: return value of '_get_actual_labels' or '_get_expected_labels'
    :return index of nonequal predA and predB
    """
    return np.where(predA != predB)[0].tolist()


def _get_index_of_eq(predA: np.ndarray, predB: np.ndarray) -> List[int]:
    """predAとpredBで同じ結果のインデックスを返す
    :param predA, predB: return value of '_get_actual_labels' or '_get_expected_labels'
    :return index of equal predA and predB
    """
    return np.where(predA == predB)[0].tolist()


def _get_NG_pred_index(model, dataset: Dataset) -> List[int]:
    """dataの予測結果のうち、labelと異なったdataのインデックスを取得する
    :param model:
    :param dataset: data and label
    :return index of mispredicted data
    """
    actual_label = _get_actual_labels(model, dataset.data)
    expected_label = _get_expected_labels(dataset.label)
    return _get_index_of_neq(actual_label, expected_label)


def _get_OK_pred_index(model, dataset: Dataset) -> List[int]:
    """dataの予測結果のうち、labelと一致したdataのインデックスを取得する
    """
    actual_label = _get_actual_labels(model, dataset.data)
    expected_label = _get_expected_labels(dataset.label)
    return _get_index_of_eq(actual_label, expected_label)


def report_NG_to_OK(modelA, modelB, cls_dataset: Dataset):
    """修正前モデルが認識できなかったが、修正後モデルが認識できるようになったものを表示
    """
    print('[NG => OK]')
    NG_pred_index = _get_NG_pred_index(modelA, cls_dataset)
    if not NG_pred_index:
        print('no data')
        return
    NG_dataset = utils.get_dataset_by_index(cls_dataset, NG_pred_index)
    OK_index_B = _get_OK_pred_index(modelB, NG_dataset)
    if not OK_index_B:
        print('no data')
        return
    plot.display_img_mnist(NG_dataset.data[OK_index_B])


def report_NG_to_NG(modelA, modelB, cls_dataset: Dataset):
    """修正前モデルが認識できず、修正後モデルも認識できなかったものを表示
    """
    print('[NG => NG]')
    NG_pred_index = _get_NG_pred_index(modelA, cls_dataset)
    if not NG_pred_index:
        print('no data')
        return
    NG_dataset = utils.get_dataset_by_index(cls_dataset, NG_pred_index)
    NG_index_B = _get_NG_pred_index(modelB, NG_dataset)
    if not NG_index_B:
        print('no data')
        return
    plot.display_img_mnist(NG_dataset.data[NG_index_B])


def report_OK_to_NG(modelA, modelB, cls_dataset):
    """修正前モデルが認識できていたが、修正後モデルが認識できなくなったものを表示
    """
    print('[OK => NG]')
    OK_pred_index = _get_OK_pred_index(modelA, cls_dataset)
    if not OK_pred_index:
        print('no data')
        return
    OK_dataset = utils.get_dataset_by_index(cls_dataset, OK_pred_index)
    NG_index_B = _get_NG_pred_index(modelB, OK_dataset)
    if not NG_index_B:
        print('no data')
        return
    plot.display_img_mnist(OK_dataset.data[NG_index_B])


def get_OK_to_NG_index(modelA, modelB, dataset:Dataset):
    """get broken data index
    :param modelA: model (before repair)
    :param modelB: model (after repair)
    :param dataset
    :return index list of broken data
    """
    OK_pred_index = _get_OK_pred_index(modelA, dataset)
    if not OK_pred_index:
        return []
    OK_dataset: Dataset = utils.get_dataset_by_index(dataset, OK_pred_index)
    NG_index_B = _get_NG_pred_index(modelB, OK_dataset)
    # もしかしたら破壊的な変更かもしれず、前つかってたやつが動かないかも。
    index = [OK_pred_index[i] for i in NG_index_B]
    return index


def get_NG_to_OK_index(modelA, modelB, dataset: Dataset):
    """get patched data index
    :param modelA: model (before repair)
    :param modelB: model (after repair)
    :param dataset
    :return index list of patched data
    """
    NG_pred_index = _get_NG_pred_index(modelA, dataset)
    if not NG_pred_index:
        return []
    NG_dataset: Dataset = utils.get_dataset_by_index(dataset, NG_pred_index)
    OK_index_B = _get_OK_pred_index(modelB, NG_dataset)
    # もしかしたら破壊的な変更かもしれず、前つかってたやつが動かないかも。
    index = [NG_pred_index[i] for i in OK_index_B]
    return index


def compare_models_pred(modelA, modelB, dataset: Dataset):
    """Compare predictions between modelA and modelB.
    :param modelA, modelB: models to compare
    :param dataset: data and label
    """
    report_func = [
        report_NG_to_OK,
        report_NG_to_NG,
        report_OK_to_NG,
    ]
    for rf in report_func:
        rf(modelA, modelB, dataset)
    print()


def compare_models_pred_each_class(modelA, modelB, dataset: Dataset):
    """modelAとmodelBの認識結果を表示する。
    - [NG => OK]: modelAで認識できなかったが、modelBで認識できたもの
    - [NG => NG]: modelAでもmodelBでも認識できなかったもの
    - [OK => NG]: modelAでは認識できたが、modelBで認識できなくなったもの
    :param modelA, modelB: 認識結果を比較したいモデル
    :param data, label: data and label (all classes)
    """
    report_func = [
        report_NG_to_OK,
        report_NG_to_NG,
        report_OK_to_NG,
    ]
    for i in range(NUM_OF_CLASS):
        print('========================')
        print('[Class:{}]'.format(i))
        print('========================')
        cls_dataset = utils.get_dataset_by_cls(dataset, i)
        try:
            for rf in report_func:
                rf(modelA, modelB, cls_dataset)
        except:
            print('No class data exists.')
        print()
        
        
def calc_repair_rate(modelA, modelB, dataset: Dataset):
    """get patched data index
    :param modelA: model (before repair)
    :param modelB: model (after repair)
    :param dataset:
    :return 
    """
    NG_pred_index_A = _get_NG_pred_index(modelA, dataset)
    if not NG_pred_index_A:
        return []
    NG_num = len(NG_pred_index_A)
    NG_dataset: Dataset = utils.get_dataset_by_index(dataset, NG_pred_index_A)
    OK_index_B = _get_OK_pred_index(modelB, NG_dataset)
    OK_num = len(OK_index_B)
    return OK_num / NG_num


def calc_broken_rate(modelA, modelB, dataset: Dataset):
    """
    """
    OK_pred_index_A = _get_OK_pred_index(modelA, dataset)
    if not OK_pred_index_A:
        return []
    OK_num = len(OK_pred_index_A)
    OK_dataset: Dataset = utils.get_dataset_by_index(dataset, OK_pred_index_A)
    NG_index_B = _get_NG_pred_index(modelB, OK_dataset)
    NG_num = len(NG_index_B)
    return NG_num / OK_num


def count_patched_and_broken(modelA, modelB, dataset: Dataset):
    """修正成功データと失敗データの数を返す.
    modelAで認識できなかったがmodelBで認識できるようになったデータ => patched
    modelAで認識できていたがmodelBで認識できなくなったデータ => broken
    :param modelA: repair target model
    :param modelB: repaired model
    :param dataset: dataset to evaluate
    :return number of patched data and broken data
    """
    patched = [] # NG => OK data
    broken = [] # OK => NG data
    for i in range(NUM_OF_CLASS):
        cls_dataset = utils.get_dataset_by_cls(dataset, i)
        patched.append(len(get_NG_to_OK_index(modelA, modelB, cls_dataset)))
        broken.append(len(get_OK_to_NG_index(modelA, modelB, cls_dataset)))

    return (patched, broken)


from decimal import Decimal, ROUND_HALF_UP
def calc_round_half_up(d, n):
    """数値dを小数点n桁で四捨五入する
    """
    round_fmt = '0.' + '0'*(n-1) + '1'
    _rounded = Decimal(d).quantize(Decimal(round_fmt), rounding=ROUND_HALF_UP)
    return _rounded


def print_rr_br(rr, br):
    """repair_rateとbreak_rateをそれぞれ四捨五入して小数点4桁まで表示する
    """
    print('{} / {}'.format(calc_round_half_up(rr, 4), calc_round_half_up(br, 4)))