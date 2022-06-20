from keras.models import Model
import numpy as np
import os
from . import utils
from . import dpatchlib
from . import plot
from .types import Dataset
from typing import NamedTuple, List


def _get_output_layer_num(model) -> int:
    """Get the number of model output layers.
    :param model:
    :returns number of output layers
    """
    # Note: If the number of output layer is 1, model.output_shape return tuple.
    return len(model.output_shape) if isinstance(model.output_shape, list) else 1


def _model_predict(model, data: Dataset.data) -> List[np.ndarray]:
    """Wrapper of 'model.predict()'.
    The purpose of this method is 
        to return a list interface results for a single output layer model.
    (The prediction result of multi output layer model is list.)
    :param model:
    :param data: data for model to predict
    :returns result (ndarray list): result of model.predict()
    """
    result = model.predict(data)
    return result if _get_output_layer_num(model) != 1 else [result]


def _calc_sum_of_outputs(outputs: List[np.ndarray]) -> List[np.ndarray]:
    return [np.sum(val, axis=0) for val in outputs]


def _calc_average_of_outputs(outputs: List[np.ndarray]) -> List[np.ndarray]:
    """Calculate average of outputs.
    :param outputs: return value of '_model_predict', or equivalent to that format.
    :return average of outputs.
    """
    return [np.mean(val, axis=0) for val in outputs]


def _to_distribution(preds: List[np.ndarray], threshold=0):
    """Convert forward values to distribution.
    :param preds: return value of '_model_predict'
    :param threshold: 発火判定の閾値（閾値以上であれば発火とみなす）
    :return distribution of forward valud of model prediction.
    """
    return [np.where(val > threshold, 1, 0) for val in preds]


def collect_average_delta_fire_nums(model, dataA, dataB, threshold=0):
    """Calc diff of number of firing by dataA and dataB.
    :param model: output model (return val of '_build_output_model')
    :param dataA, dataB: 差分値を見たいデータ
    :param threshold: 発火判定の閾値
    :return output_average_delta_value (float list): 各層の発火平均回数の差分値
    """
    def _calc_average_distribution(data):
        preds = _model_predict(model, data)
        dists = _to_distribution(preds, threshold=threshold)
        average_dists = _calc_average_of_outputs(dists)
        return average_dists

    average_dist_A = _calc_average_distribution(dataA)
    average_dist_B = _calc_average_distribution(dataB)

    return [a - b for a, b in zip(average_dist_A, average_dist_B)]


def _build_output_model(model, target_layer: List[int]):
    """Build model of outputs specified target layer.
    :param model: keras model
    :param target_layer: 出力させたい層のインデックスリスト
    """
    keras_layer = [model.layers[tl] for tl in target_layer]
    output_layers = [kl.output for kl in keras_layer]
    output_model = Model(inputs=model.inputs, outputs=output_layers)
    return output_model


def get_delta_value(
        model,
        target_layer: List[int],
        target_val_data, target_adv_data,
        fire_threshold=0):
    """2種類のデータを受け取り、target_layerで指定した層における出力の差分を取得する
    :param model: keras model
    :param target_layer: 対象とする層のインデックス
    :param target_val_data, target_adv_data: 正常系と異常系のデータ。同じクラスのものを与えること。
    :param fire_threshold: 発火判定の閾値
    :return delta_value: 各層の出力差分値
    """
    output_model = _build_output_model(model, target_layer)

    delta_value = collect_average_delta_fire_nums(
        output_model,
        target_val_data,
        target_adv_data,
        fire_threshold)

    return delta_value

def get_delta_value_dict(
        model,
        target_layer_indices: List[int],
        pos_data: Dataset.data,
        neg_data: Dataset.data,
        fire_threshold=0):
    """
    """
    delta_value = get_delta_value(model,
                                  target_layer_indices,
                                  pos_data,
                                  neg_data,
                                  fire_threshold)
    return dict(zip(target_layer_indices, delta_value))

def calculate_repair_results(
        model,
        target_layer_index: List[int],
        percentages: List[float],
        fix_rates: List[float],
        delta_value,
        val_dataset: Dataset,
        adv_dataset: Dataset,
        delta_index: List[List[int]],
        repair_function):
    """
    percentagesとfix_ratesによってどのように修正が変化するか調べる。
    :param model: 修正対象のモデル
    :param validation_model: modelと同じ構造を持つモデル。
    :param target_layer_index (int List): 修正対象層のインデックス
    :param percentages (float List): 閾値のリスト
    :param fix_rates (float List): 調査したい修正率のリスト
    :param delta_value: 差分値
    :param x_norm, y_norm: 修正後モデルの評価に使用
    :param x_adv, y_adv: 修正後モデルの評価に使用
    :param delta_index: 
    :param repair_function: function of repairing model weights.
    """
    num_classes = 10
    val_results = []
    adv_results = []
    
    val_model = utils.copy_model(model)

    count = 1
    for p in percentages:
        each_val_result = []
        each_adv_result = []
        for fr in fix_rates:
            w = repair_function(model, target_layer_index, delta_value, p, fr)
            val_model.set_weights(w)
            _evaluate = lambda dataset: val_model.evaluate(dataset.data, dataset.label, verbose=0)

            # 各数字ごとの認識率
            val_cls_results = []
            adv_cls_results  = []
            for i in range(num_classes):

                _val_dataset = utils.get_dataset_by_cls(val_dataset, i)
                _adv_dataset = utils.get_dataset_by_index(adv_dataset, delta_index[i])
                
                score_val = _evaluate(_val_dataset)
                score_adv = _evaluate(_adv_dataset)

                if score_val == []:
                    score_val = [0.0, 0.0]
                if score_adv == []:
                    score_adv = [0.0, 0.0]
                val_cls_results.append(score_val[1])
                adv_cls_results.append(score_adv[1])

            each_val_result.append(val_cls_results)
            each_adv_result.append(adv_cls_results)
            print("excuting... {}/{}".format(count, len(percentages)*len(fix_rates) ), "\r", end="")
            # print("*", end="")
            count += 1
        val_results.append(each_val_result)
        adv_results.append(each_adv_result)

    return val_results, adv_results


def summarize_repair_process(
        model,
        repair_target_layer: List[int],
        repair_target_class: int,
        val_dataset: Dataset,
        adv_dataset: Dataset,
        fire_threshold=0):
    """修正範囲割合、修正率を10%刻みで変化させた場合の結果を表示する。
    repair_target_layerやfire_thresholdによって修正効果が変わるかどうかを調べるために使用。
    :param model: model to repair
    :param repair_target_layer: index of repair target layer
    :param repair_target_class: repair target class
    :param val_dataset: validation data
    :param adv_dataset: repair target data
    :param fire_threshold: threshold value of fire.
    """
    # Get data of repair target
    repair_target_val_data = utils.get_dataset_by_cls(
        val_dataset, repair_target_class).data
    mispred_indices = utils.get_indices_of_mispred(model, adv_dataset)
    repair_target_adv_data = utils.get_dataset_by_index(
        adv_dataset, mispred_indices[repair_target_class]).data

    # Calculate delta value
    delta_value = get_delta_value(
        model,
        repair_target_layer,
        repair_target_val_data, 
        repair_target_adv_data,
        fire_threshold=fire_threshold)

    # Set parameters in 10% increments.
    percentages_for_summary = [x/100 for x in range(10, 105, 10)]
    repair_rates_for_summary = [x/100 for x in range(0, 101, 10)]

    # Calc repair result.
    val_results, adv_results = calculate_repair_results(
        model,
        repair_target_layer,
        percentages_for_summary,
        repair_rates_for_summary,
        delta_value,
        val_dataset,
        adv_dataset,
        mispred_indices,
        dpatchlib.get_fixed_weights_ver4)

    # Show result.
    plot.plot_result_after_fix(
        val_results,
        adv_results,
        percentages_for_summary,
        repair_rates_for_summary,
        repair_target_class)

    return;


class Metrics(NamedTuple):
    neg_average: float
    neg_unit_rate: float
    neg_score: float


def calc_neg_metrics(dv):
    """1つの層の差分値に対してメトリクスを算出する。
        - neg_average: 差分値のうち、負値の平均値
        - neg_unit_rate: 差分値のうち、負値となったunitの数
        - neg_score: neg_average と neg_unit_rate の積
    :param dv: return value of 'get_delta_value'
    :return neg_metrics: 差分値から算出したメトリクス
    """
    dv_unit_num = dv.size
    neg_dv = dv[dv < 0]
    neg_dv_unit_num = neg_dv.size
    neg_average = neg_dv.sum() / neg_dv_unit_num if neg_dv_unit_num != 0 else 0
    neg_dv_unit_rate = neg_dv_unit_num/dv_unit_num * 100
    print('[calc_neg_metrics] neg_num: {}  all_num: {}'.format(neg_dv_unit_num, dv_unit_num))
    neg_score = neg_average * neg_dv_unit_rate
    return Metrics(neg_average, neg_dv_unit_rate, neg_score)


def rank_repair_target_layer(
        model,
        target_layer,
        target_val_data, target_adv_data,
        fire_threshold=0,
        verbose=False):
    """calc_neg_metricsによって算出したneg_scoreを元に修正対象層をソートする。
    neg_scoreをソートに利用する。
    :param model: keras model
    :param target_layer: 修正対象層のインデックス
    :param target_x_val_norm, target_x_val_adv: 同クラスの正常系データと異常系データ
    :param fire_threshold: 差分値計算時の発火判定閾値。（0がよいと思われる。）
    :param verbose: Trueをセットすると、その他メトリクスをあわせてprintする。
    :return neg_scoreの小さい順（絶対値の大きい順）にソートした、修正対象層のインデックスリスト
    """
    delta_value = get_delta_value(
        model,
        target_layer,
        target_val_data, target_adv_data,
        fire_threshold=fire_threshold)

    res = []
    for layer_index, dv in zip(target_layer, delta_value):
        metrics = calc_neg_metrics(dv)
        res.append((layer_index, metrics.neg_score))
        if verbose:
            print('[index of layer: {}]'.format(layer_index))
            print('neg_average: {:.3f}'.format(metrics.neg_average))
            print('neg_unit_rate: {:.2f}%'.format(metrics.neg_unit_rate))
            print('neg_score: {:.3f}'.format(metrics.neg_score))
            print()

    # 'neg_score'の小さい順（絶対値が大きくなる順）にsort
    res.sort(key=lambda x: x[1])
    return res


def rank_repair_target_class(model, val_dataset: Dataset, adv_dataset: Dataset):
    """Rank repair target classes.
    正常系データの認識率が高く、異常系データの認識率が低いものを高く評価する。
    （正常系データの認識率が低い場合はそもそも修正効果があまりないと考えられるため。）
    :param model: 修正対象モデル
    :param val_dataset: validation dataset
    :param adv_dataset: repair target dataset
    :return repair_taget_class: 修正対象クラスをランク付けしたもの。
    """
    val_scores = utils.get_result_score(model, val_dataset)
    adv_scores = utils.get_result_score(model, adv_dataset)

    repair_target_class = []
    for i in range(10):
        label = str(i)
        repair_target_class.append((label, val_scores[label]*(1-adv_scores[label])))

    repair_target_class.sort(key=lambda x: x[1], reverse=True)
    return repair_target_class
