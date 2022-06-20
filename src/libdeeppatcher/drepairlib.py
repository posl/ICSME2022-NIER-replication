import keras
import numpy as np
from . import utils
from . import measure
from typing import List, Any, Dict, NamedTuple, Tuple
from .types import Dataset, PatchParam, MatrixId


class PatchResult(NamedTuple):
    score: float
    layer_index: int
    fire_threshold: float
    patch_range_rate: float
    decrement_rate: float


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


def _get_negative_values(delta_value) -> np.ndarray:
    """Get only negative values from delta_value
    :param delta_value:
    :return negative values of delta_value
    """
    negative_all_vals = np.array([])
    # delta_valueから値がminusになったものだけを抜き出す。
    # 畳み込みの場合は、マイナスの和の平均に変換
    for dv in delta_value.values():
        if len(dv.shape) == 3:
            for i in range(dv.shape[2]):
                dv_flatten = np.concatenate(dv)[:, i]
                neg_vals = dv_flatten[np.where(dv_flatten < 0)]
                total_neg_val = np.sum(neg_vals)
                num_of_neg_vals = neg_vals.shape[0]
                negative_all_vals = np.concatenate([negative_all_vals,
                                                    np.array([total_neg_val/num_of_neg_vals])])
        else:
            neg_vals = dv[np.where(dv < 0)]
            negative_all_vals = np.concatenate([negative_all_vals, neg_vals])
    return negative_all_vals


def _get_val_of_top_of_x_percent(negative_values, percentage):
    """
    :param negative_values:
    :param percentage:
    """
    # Sort in order of absolute value (decending negative value)
    _sorted_indices = np.argsort(negative_values).tolist()
    index = int(len(_sorted_indices) * percentage) - 1 # 0-index
    val = negative_values[_sorted_indices[index]] if index != -1 else -10**3
    return val


def _calc_threshold(delta_value, percentage):
    """delta_valueからマイナス値のみを取り出し、percentageで指定した上位x%に当たる出力値を計算
    :param delta_value:
    :param percentage: 上位x%を指定
    :return threshold: percentageで指定した上位x%に当たる値
    """
    negative_all_vals = _get_negative_values(delta_value)
    threshold = _get_val_of_top_of_x_percent(negative_all_vals, percentage)
    return threshold


def _localize(delta_value: Dict[int, np.ndarray], threshold) -> Dict[int, List[int]]:
    index = {}
    for layer_index, dv in delta_value.items():
        if len(dv.shape) == 3:
            idx = []
            for j in range(dv.shape[2]):
                dv_flatten = np.concatenate(dv)[:, j]
                neg_vals = dv_flatten[np.where(dv_flatten < 0)]
                num_of_neg = neg_vals.shape[0]
                average_neg_val = np.sum(neg_vals) / num_of_neg
                if average_neg_val <= threshold:
                    idx.append(j)
            index[layer_index] = idx
        else:
            idx = np.where(dv <= threshold)[0].tolist()
            index[layer_index] = idx
    return index


def localize(
        model,
        target_layer_indices: List[int],
        pos_dataset: Dataset,
        neg_dataset: Dataset,
        fire_threshold: float,
        repair_range_ratio) -> Dict[int, List[int]]:
    """Localize neurons based on fire_threshold and repair_range_ratio
    :param model:
    :param target_layer_indices: layer indices to repiar
    :param pos_dataset:
    :param neg_dataset:
    :param fire_threshold: set threshold for neuron firing
    :param repair_range_ratio: decide to repair the neurons in the top x%
    """
    delta_value = measure.get_delta_value_dict(model, target_layer_indices,
                                               pos_dataset.data, neg_dataset.data,
                                               fire_threshold=fire_threshold)
    boundary_val = _calc_threshold(delta_value, repair_range_ratio)
    localized_neuron_indices = _localize(delta_value, boundary_val)
    return localized_neuron_indices


def patch(model,
          repair_target_indices: Dict[int, List[int]],
          decrement_rate: float):
    """Generate patched weights.
    :param model
    :param repair_target_indices: localizedf neuron indices (return value of 'localize')
    :param decrement_rate: how much to reduce the weights
    :return patched weights
    """
    weights = [layer.get_weights() for layer in model.layers]
    for layer_index, indices in repair_target_indices.items():
        target_weights = weights[layer_index]
        # target_weights[0] => kernel
        # target_weights[1] => bias
        # This repair targets only kernel.
        if target_weights == []:
            # Skip layer without params
            continue
        if len(target_weights[0].shape) == 4:
            # 畳み込み層: weights[layer_index][0] => (3,3,1,12) のような形式
            for i in indices:
                target_weights[0][:,:,:,i] *= (1 - decrement_rate)
        else:
            # 全結合層の場合 weights[layer_index][0] => (1568, 200)のような形式
            decrement_matrix = np.eye(target_weights[0].shape[1], dtype='float32')
            for i in indices:
                decrement_matrix[i][i] = (1 - decrement_rate)
            weights[layer_index][0] = np.dot(target_weights[0], decrement_matrix)
    return _convert_weights_for_keras(weights)


def repair(
        model,
        pos_dataset: Dataset,
        neg_dataset: Dataset,
        patch_params: PatchParam):
    """
    """
    target_layer_indices = patch_params.layer_indices
    fire_threshold = patch_params.fire_threshold
    patch_range_rate = patch_params.patch_range_rate
    decrement_rate = patch_params.decrement_rate
    
    localized_neuron_indices = localize(model, target_layer_indices,
                                        pos_dataset, neg_dataset,
                                        fire_threshold,
                                        patch_range_rate)
    patched_weights = patch(model, localized_neuron_indices, decrement_rate)
    return patched_weights


def run_all_repair(
        model,
        pos_dataset: Dataset,
        neg_dataset: Dataset,
        degrade_limit: float,
        target_layer_indices,
        percentages: List[float],
        decrement_rates: List[float],
        fire_threshold: float = 0.0):
    """
    """
    # Save original model wegihts to restore after patching
    original_weights = model.get_weights()

    each_pos_dataset = [utils.get_dataset_by_cls(pos_dataset, i)
                        for i in range(10)]

    first_scores = [model.evaluate(*cls_dataset, verbose=0)[1]
                    for cls_dataset in each_pos_dataset]

    def _is_degrade(patched_model) -> bool:
        """If patched_model is degraded, return True
        """
        for first_score, cls_dataset in zip(first_scores, each_pos_dataset):
            patched_score = patched_model.evaluate(*cls_dataset, verbose=0)[1]
            if (first_score - patched_score) > degrade_limit:
                return True
        return False

    repair_results = []
    for p in percentages:
        print('\npercentage: {}'.format(p))
        ## localize
        for i in range(11):
            fire_val = fire_threshold + i/10
            localized_neuron_indices = localize(model, target_layer_indices,
                                                pos_dataset, neg_dataset,
                                                fire_val,
                                                p)

            # if localized neurons do not exists, skip to patch.
            # => もしニューロンが空だったらfire_thresholodを増加する処理やっても良さげ。（マックス1.0まで？）
            flatten = []
            for indices in localized_neuron_indices.values():
                flatten += indices

            no_localized = True
            if flatten == []:
                if i == 10:
                    break
                print('No localized neurons. fire threshold: {} '.format(fire_val))
                continue
            else:
                no_localized = False
                break

        if no_localized:
            print('No localized neurons. Skip patch.')
            continue

        for dr in decrement_rates:
            ## patch
            model.set_weights(original_weights)
            patched_weights = patch(model, localized_neuron_indices, dr)
            model.set_weights(patched_weights)
            if _is_degrade(model):
                break
            patched_score = model.evaluate(*neg_dataset, verbose=0)[1]

            ## Save repair result
            repair_results.append(PatchResult(
                patched_score,
                target_layer_indices[0],
                fire_val,
                p,
                dr))
            print('*', end='')

    ## Restore original model
    model.set_weights(original_weights)
    return repair_results


def get_patched_weights(
        model,
        repair_target_dataset: Tuple[MatrixId, Dataset],
        pos_dataset: Dataset,
        result_index: int = 0,
        results_path='./results/drepair_',):
    """修正結果から修正パッチ（重み）を取得する
    :param model: origin model
    :param results_path: 修正結果ログが格納されたディレクトリ
    :param repair_target_dataset: 修正対象データセット
    :param pos_dataset: 修正時に使用したvalidation dataset
    :param result_index: 修正結果のうち上位から何番目を復元するか。(defaultはtop)
    :return patched_weight: 
    """
    matrix_id, neg_dataset = repair_target_dataset
    patch_results = pd.read_csv(
        results_path \
        + str(matrix_id.expected_label) + '_' \
        + str(matrix_id.predicted_label) + '.csv'
    )
    results_sorted = patch_results.sort_value(
        ['score', 'patch_range_rate', 'decrement_rate'],
        ascending=[False, True, True]
    )
    best_result = results_sorted.iloc[result_index]
    lyr_id = [int(best_result.layer_index)]
    ft = best_result.fire_threshold
    prr = best_result.patch_range_rate
    dr = best_result.decrement_rate
    
    patched_weight = repair(
        model,
        pos_dataset,
        neg_dataset,
        PatchParam(lyr_id, ft, prr, dr)
    )
    return patched_weight