import keras
from keras.models import load_model, Model
import numpy as np
from . import utils
from . import measure
from typing import List, Any
from .types import Dataset

def _convert_weights_for_keras(weights: List[List[Any]]):
    """Convert weights to keras format.
    :param weights: weight parameters
    :return weights_for_keras: weights formated with keras.
    """
    weights_for_keras = []
    for weight in weights:
        for w in weight:
            weights_for_keras.append(w)
    return weights_for_keras


def get_fixed_weights_ver4(
        model,
        target_layer,
        delta_value,
        percentage=1.0,
        decrement_rate=0):
    """畳み込み層も含めて修正を行う。あとはver3と同じ
    :param model  (keras_model): 修正対象とするモデル
    :param target_layer (int list): 修正対象の層
    :param delta_value (ndarray): 差分の値
    :param percentage (int): 差分の値からどの部分を修正するか決める閾値
    :param decrement_rate (int): 重み修正の割合。0~1
    :return weights (ndarray): 修正したモデルの重み
    """
    ## 閾値の設定
    total_delta = np.array([])
    for d in delta_value:
        # 畳み込みの場合
        if len( d.shape ) == 3:
            for i in range(d.shape[2]):
                a = np.concatenate(d)[:, i]
                b = a[np.where(a < 0)]
                len_b = b.shape[0]
                total_b = np.sum(b)
                total_delta = np.concatenate([
                    total_delta,
                    np.array([total_b/len_b])
                ])
        # 全結合層の場合
        else:
            a = d[np.where(d < 0)]
            total_delta = np.concatenate([total_delta, a])

    _sorted = np.argsort(total_delta).tolist()
    partition = int(len(_sorted) * percentage) - 1

    ## 閾値の決定
    threshold = total_delta[_sorted[partition]] if not partition == -1 else -10**3

    index = {} # 修正するニューロンのインデックス
    for i, d in enumerate(delta_value):
        # 畳み込み層の場合
        if len(d.shape) == 3:
            idx = []
            for j in range(d.shape[2]):
                a = np.concatenate(d)[:, j]
                b = a[np.where(a < 0)]
                len_b = b.shape[0]
                ave_b = np.sum(b) / len_b
                if ave_b < threshold:
                    idx.append(j)
            index[target_layer[i]] = idx
        # 全結合層の場合
        else:
            idx = np.where(delta_value[i] < threshold)[0].tolist()
            index[target_layer[i]] = idx

    # モデルの重みを取り出す。（取り出した重みを修正する）
    weights = []
    for i in range(len(model.layers)):
        weights.append(model.layers[i].get_weights())

    # 重みの修正
    # [layer][0] の[0]は、重みパラメータ（wx + bにおけるw）を表し、[1]であればwx + bのbを表す。
    # bは修正対象外
    for layer in target_layer:
        # 畳み込み層: weights[layer][0] => (3, 3, 1, 12)のような形式となる
        if weights[layer] == []:
            continue
        if len(weights[layer][0].shape) == 4:
            for i in index[layer]:
                weights[layer][0][:,:,:,i] *= (1 - decrement_rate)
        # 全結合層の場合: weights[layer][0] => (1568, 200)のような形式となる
        else:
            fix_matrix = np.eye(weights[layer][0].shape[1], dtype='float32')
            for i in index[layer]:
                fix_matrix[i][i] = (1 - decrement_rate)
            weights[layer][0] = np.dot(weights[layer][0], fix_matrix)

    # kears形式に変換する.
    fixed_weights = _convert_weights_for_keras(weights)
    return fixed_weights


def get_best_repair_param(
        model,
        target_layer,
        percentages, decrement_rates,
        delta_value,
        boundary_score,
        val_dataset: Dataset,
        repair_target_dataset: Dataset,
        repair_weights):
    """Return a param of percentage and decrement_rate of best repair.
    :param model: model to repair
    :param percentages (double list): 修正範囲を決める値
    :param decrement_rates (double list): 修正率を決める値
    :param delta_value: 差分値
    :param boundary_score: 最低限 val_dataに対して保証するためのスコア
    :param val_dataset: validation data and label
    :param repair_target_dataset: repair target dataset of one class 
    :param repair_weights: 重み修正を行う関数
    :returns (p, r): 最も良いスコアとなった修正範囲と修正率の値
    """
    num_classes = 10
#     validation_model = utils.copy_model(model)
    original_weights = model.get_weights()

    each_val_dataset = [utils.get_dataset_by_cls(val_dataset, i)
                        for i in range(num_classes)]

    # 修正前の検証用データのスコア
    first_score = [model.evaluate(*cls_dataset, verbose=0)[1]
                   for i, cls_dataset in enumerate(each_val_dataset)]

    def _is_degrade(val_model) -> bool:
        for i, cls_dataset in enumerate(each_val_dataset):
            val_score = val_model.evaluate(*cls_dataset, verbose=0)[1]
            if (first_score[i] - val_score > boundary_score):
                return True
        return False

    param_candidates = {}
    for p in percentages:
        min_score = 1.1
        print("\npercentage: {}".format(p))
        for fr in decrement_rates:
            model.set_weights(original_weights)
            w = repair_weights(model, target_layer, delta_value, p, fr)
            model.set_weights(w)
            # Check model degration.
            if _is_degrade(model):
                break
            score_adv = model.evaluate(*repair_target_dataset,
                                                  verbose=0)[1]
            param_candidates[(p, fr)] = score_adv
            print("*", end="")

    ## Restore original model
    model.set_weights(original_weights)
    best_param = max(param_candidates, key=param_candidates.get)
    return best_param

def repair_model(
        model,
        target_layer,
        target_class,
        val_dataset: Dataset,
        repair_target_dataset: Dataset,
        degration_limit=0.03,
        fire_threshold=0):
    """
    :param model: 修正対象モデル
    :param target_layer (int List): 修正対象層のインデックス
    :param degration_limit: 修正によるデグレーションの上限値
    :param target_class: 修正対象のクラス（target_x_val_advとクラスを一致させること）
    :param x_val_norm, y_val_norm: 正常系のデータ（全てのクラス）
    :param target_x_val_adv, target_y_val_adv: 修正対象のデータ（target_classで指定するクラス）
    :param fire_threshold: 発火判定の閾値（差分値の計算に使用）

    :return repaired model
    """
    # Calc delta fire-nums
    output_model = measure._build_output_model(model, target_layer)
    target_val_dataset = utils.get_dataset_by_cls(val_dataset, target_class)    
    delta_value = measure.collect_average_delta_fire_nums(
        output_model,
        target_val_dataset.data,
        repair_target_dataset.data,
        fire_threshold)

    # Repair
    percentages = [x/100 for x in range(5, 105, 5)]
    repair_rates = [x/100 for x in range(1, 101, 1)]

    best_param = get_best_repair_param(
        model,
        target_layer,
        percentages, repair_rates,
        delta_value,
        degration_limit,
        val_dataset,
        repair_target_dataset,
        get_fixed_weights_ver4)

    p, rr = best_param
    print()
    print(f'REPAIR RESULT\npercentage: {p} repair-rate: {rr}')

    repaired_weights = get_fixed_weights_ver4(
        model,
        target_layer,
        delta_value,
        percentage=p,
        decrement_rate=rr)

    repaired_model = utils.copy_model(model)
    repaired_model.set_weights(repaired_weights)

    return repaired_model

