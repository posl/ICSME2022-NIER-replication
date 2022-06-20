import numpy as np
import pandas as pd
import math
from .types import Dataset, MatrixId
from typing import List, Tuple
from . import report

def _log2(x):
    try:
        return math.log2(x)
    except ValueError:
        return -math.inf

def calc_entropy(pred):
    """calculate entropy of model prediction
    :param pred: output of model prediction
    :return entropy
    """
    return -sum([p*_log2(p) if p != 0 else 0 for p in pred[0]])


def get_index_high_entropy(predicts, t=0.1):
    """エントロピーが一定以上となるデータのインデックスを取得する
    :param predicts: output of model predction
    :param t: if entropy is above t, entropy is high.
    how to use
    ===
    get_index_high_entropy(model.predict(x), t=0.01)
    ===
    """
    return [
        i for i, prob in enumerate(predicts)
        if calc_entropy(prob) >= t
    ]


def get_index_low_entropy(predicts, t=0.1):
    """エントロピーが一定未満となるデータのインデックスを取得する
    :param predicts: output of model prediction
    :param t: if entropy is below t, entropy is low.
    how to use
    ===
    get_index_low_entropy(model.predict(x), t=0.01)
    ===
    """
    return [
        i for i, prob in enumerate(predicts)
        if calc_entropy(prob) < t
    ]


def get_entropy(model, data): 
    """1件のデータのentropyを計算
    """
    pred = model.predict(data[0:1])
    entropy = calc_entropy(pred)
    return entropy


def get_pcs(model, data):
    """1件のデータのpcsを計算
    """
    pred = model.predict(data[0:1])
    top2_score, top1_score = np.partition(pred.flatten(), -2)[-2:]
    return top1_score - top2_score


def get_label_pred_score(model, data, label):
    """1件のデータのlabel_predを計算
    """
    label_cls = label.argmax()
    label_pred = model.predict(data, verbose=0)[0][label_cls]
    return label_pred


def get_loss(model, data, label):
    label_cls = label.argmax()
    loss = model.evaluate(data, label, verbose=0)[0]
    return loss


def calc_metrics_one_data(model, data, label):
    """Calculating metrics to one data
    :param model:
    :param data: one data
    :param label: label of data
    ---
    pcs: predicted confidence score
    entropy: entropy
    label_pred_score: 正解クラスへの予測スコア
    predicted_cls: 予測されたクラス（label_clsと異なっている場合、予測を間違ったということ）
    data: dataそのもの
    label_cls: dataの正解クラス
    result: 修正結果（成功・失敗）を表す。初期値は0
    ---
    :return the score of above metrics
    """
    pcs = get_pcs(model, data)
    entropy = get_entropy(model, data)
    label_pred_score = get_label_pred_score(model, data, label)
    loss = get_loss(model, data, label)
    predicted_cls = model.predict(data, verbose=0).argmax()
    label_cls = label.argmax()
    res = {
        'pcs': pcs,
        'entropy': entropy,
        'label_pred_score': label_pred_score,
        'loss': loss,
        'predicted_cls': predicted_cls,
        'data': data,
        'label': label,
        'label_cls': label_cls,
        'result': 0,
    }
    return res


def calc_metrics(model, dataset: Dataset):
    """datasetの全てのデータに対して、それぞれmetricsを算出する
    :param model: metrics算出対象のモデル
    :param dataset: metrics算出対象のデータセット
    """
    result = {}
    for i in range(dataset.data.shape[0]):
        data, label = dataset.data[i:i+1], dataset.label[i:i+1]
        result[i] = calc_metrics_one_data(model, data, label)
    return result


def _get_patched_weight_name(model_name: str, layer_id: int, matrix_id: MatrixId, iter_count: int, data_type=None):
    """修正された重みのpathを取得する
    :param model_name: 修正対象モデルの名前
    :param layer_id: 修正対象layerのインデックス
    :param matrix_id: 修正対象クラスのmatrix_id
    :param iter_count: 何回目の修正適用かどうか(0~4の計5回のうち)
    :param 修正適用重みへのパス
    """
    if data_type is not None:
        patched_weights_name = \
            './results/' + model_name + '_arachne_weights_' \
            + data_type + '_' + \
            str(matrix_id.expected_label) + '_' + str(matrix_id.predicted_label) + '_iter_' + str(iter_count) + '.h5'
        return patched_weights_name

    patched_weights_name = \
        './results/' + model_name + '_arachne_weights_' + \
        'layer' + str(layer_id) + '_' + \
        str(matrix_id.expected_label) + '_' + str(matrix_id.predicted_label) + '_iter_' + str(iter_count) + '.h5'
    return patched_weights_name


def calc_metrics_of_dataset_average(
    origin_model,
    tmp_model,
    model_name,
    repair_target_dataset: Tuple[MatrixId, Dataset],
    metrics_target_dataset: Dataset,
    layer_id,
    data_type=None):
    """metrics_target_datasetに対してメトリクスを計算し、
    修正後のモデルが認識できるようになったか、もしくは認識できなくなったか結果をまとめたDataFrameを返す。
    :param origin_model: 修正対象モデル
    :param tmp_model: 修正適用のための一時的なモデル
    :param repair_target_dataset: 修正対象データセット(1クラス)
    :param metrics_target_dataset: metricsを調べたいデータセット
    :return metrics_target_datasetへのmetricsの評価結果
    """
    matrix_id, neg_dataset = repair_target_dataset
    result = calc_metrics(origin_model, metrics_target_dataset)
    for i in range(5):
        patched_weights_name = _get_patched_weight_name(model_name, layer_id, matrix_id, i, data_type=data_type)
        tmp_model.load_weights(patched_weights_name)
        for i in report.get_NG_to_OK_index(origin_model, tmp_model, metrics_target_dataset):
            if 'success_num' not in result[i].keys():
                result[i]['success_num'] = 1
            else:
                result[i]['success_num'] += 1
        for i in report.get_OK_to_NG_index(origin_model, tmp_model, metrics_target_dataset):
            if 'success_num' not in result[i].keys():
                result[i]['success_num'] = -1
            else:
                result[i]['success_num'] -= 1
    df = pd.DataFrame.from_dict(result, orient='index')
    return df


def aggregate_results_of_arachne(
        origin_model,
        tmp_model,
        model_name,
        repair_target_datasets: List[Tuple[MatrixId, Dataset]],
        layer_id,
        data_type=None):
    """arachneによる修正結果として、修正対象データセットの修正成功、失敗をまとめたDataFrameを返す
    :param origin_model: 修正対象モデル
    :param tmp_model: パッチを適用するための仮モデル
    :param model_name: 実験対象モデルの名前(mnist_dense, mnist_convolutional, resnet20)
    :param repair_target_datasets: 修正対象データセット群
    :param layer_idx: どの層を修正するか
    """
    results = []
    for matrix_id, neg_dataset in repair_target_datasets:
        print(matrix_id)
        print('num:', neg_dataset.data.shape[0])
        print(origin_model.evaluate(*neg_dataset, verbose=0))
        result = calc_metrics(origin_model, neg_dataset)
        for v in result.values():
            v['success_num'] = 0
        for i in range(5):
            patched_weights_name = _get_patched_weight_name(model_name, layer_id, matrix_id, i, data_type=data_type)
            tmp_model.load_weights(patched_weights_name)
            for i in report.get_NG_to_OK_index(origin_model, tmp_model, neg_dataset):
                result[i]['success_num'] += 1
        results.append(result)
    df_lst = [pd.DataFrame.from_dict(r, orient='index') for r in results]
    _df_total = pd.concat(df_lst)
    df_total = _df_total.reset_index(drop=True)
    return df_total


def calc_average_rate(
        origin_model,
        tmp_model,
        model_name,
        matrix_id,
        dataset,
        layer_id,
        calc_rate_func):
    """matrix_idで指定した修正結果における、datasetに対するrepair_rateもしくはbreak_rateを計算する.
    5回の修正結果の平均を算出する。
    :param origin_model: 修正対象モデル
    :param tmp_model: 修正を適用するための一時的なモデル
    :param model_name: 修正対象モデルの名前
    :param matrix_id: 修正対象クラスのmatrix_id
    :param dataset: rateを計算したいデータセット
    :param layer_id: 修正対象層のインデックス
    :param calc_rate_func: 
        `report.calc_repair_rate` or `report.calc_broken_rate`を渡す
    """
    sum_rate = 0
    for i in range(5):
        patched_weights_name = _get_patched_weight_name(model_name, layer_id, matrix_id, i)
        tmp_model.load_weights(patched_weights_name)
        sum_rate += calc_rate_func(origin_model, tmp_model, dataset)
    return sum_rate / 5
