from .types import Dataset, WeightId, HitSpectrum, SuspiciousnessScore
from . import measure
from typing import List, Dict
import numpy as np
from keras import backend as K
import math


def _tarantula(hit_spectrum: HitSpectrum):
    _as = hit_spectrum.activate_pos
    _af = hit_spectrum.activate_neg
    _ns = hit_spectrum.not_activate_pos
    _nf = hit_spectrum.not_activate_neg
    suspiciousness = (_af/(_af+_nf)) / ((_af/(_af+_nf)) + (_as/(_as+_ns)))
    if np.isnan(suspiciousness):
        suspiciousness = 0
    return suspiciousness


def _ochiai(hit_spectrum: HitSpectrum):
    _as = hit_spectrum.activate_pos
    _af = hit_spectrum.activate_neg
    _ns = hit_spectrum.not_activate_pos
    _nf = hit_spectrum.not_activate_neg
    suspiciousness = _af / math.sqrt((_af+_nf)*(_af+_as))
    if np.isnan(suspiciousness):
        suspiciousness = 0
    return suspiciousness


def _compute_hit_spectrums(
        model,
        target_layer: List[int],
        pos_dataset: Dataset,
        neg_dataset: Dataset,
        threshold=0) -> Dict[int, np.float64]:
    """hit_spectrumsを算出する
    :param model
    :param target_layer: 算出対象の層のインデックス
    :param pos_dataset: 正常系データセット
    :param neg_dataset: 異常系データセット
    :param threshold:
        発火のしきい値.
        特に最終層は0だと全てが発火することになるので、0より大きい値をセットする必要がある
    """
    # 各ニューロンのHitSpectrumを計算
    output_model = measure._build_output_model(model, target_layer)
    def _calc_distribution(data):
        preds = measure._model_predict(output_model, data)
        _dists = measure._to_distribution(preds, threshold=threshold)
        dists = measure._calc_sum_of_outputs(_dists)
        return dists
    def _calc_hit(data):
        hit = _calc_distribution(data)
        no_hit = [data.shape[0] - h for h in hit]
        return hit, no_hit
    _pos_hit, _pos_no_hit = _calc_hit(pos_dataset.data)
    _neg_hit, _neg_no_hit = _calc_hit(neg_dataset.data)
    hit_spectrums = {}
    for i, layer in enumerate(target_layer):        
        hit_spectrum = [
            HitSpectrum(pa, pn, na, nn)
            for pa, pn, na, nn
            in zip(_pos_hit[i], _pos_no_hit[i], _neg_hit[i], _neg_no_hit[i])
        ]
        hit_spectrums[layer] = hit_spectrum
    return hit_spectrums


def _to_suspiciousness_scores(hit_spectrums: Dict[int, List[HitSpectrum]], approach='tarantula'):
    """HitSpectrumを疑惑値に変換する
    !! use `_compute_suspiciousness_scores` instead of this function
    :param hit_spectrums
    :return suspiciousness_scores
    """
    approachs = {
        'tarantula': _tarantula,
        'ochiai': _ochiai,
    }
    calc_susp_score = approachs[approach]

    suspiciousness_scores = {}
    for layer, spectrums in hit_spectrums.items():
        suspiciousness = [calc_susp_score(hs) for hs in spectrums]
        suspiciousness_scores[layer] = suspiciousness
    return suspiciousness_scores


def _to_suspicious_distribution(suspiciousness_scores: Dict[int, np.float], k: int=30):
    """複数層が対象の場合に、層ごとの疑惑度を算出する。
    複数層がターゲットの際に、どの層の疑惑度が高いかを決めるためのものなので、
    1つの層が対象の場合、数値はでるが意味をなさないことに注意。
    :param suspiciousness_scores: return val of `_to_suspiciousness_scores`
    :param k: 疑惑値の高いニューロン数の上位何件を対象にするかを決める
    :return susp_distribution: 各層の疑惑度
    """
    neuron_nums_of_layer = {}
    all_suspiciousness = []
    for layer, susp_scores in suspiciousness_scores.items():
        neuron_nums_of_layer[layer] = len(susp_scores)
        all_suspiciousness += [
            SuspiciousnessScore(layer, i, susp_score)
            for i, susp_score
            in enumerate(susp_scores)
        ]
    all_suspiciousness.sort(key=lambda x: x.suspiciousness, reverse=True)
    all_suspiciousness = all_suspiciousness[:k]
    
    susp_distribution = {layer: 0 for layer in suspiciousness_scores.keys()}
    for susp_score in all_suspiciousness:
        susp_distribution[susp_score.layer_index] += 1
    for layer, neuron_nums in neuron_nums_of_layer.items():
        susp_distribution[layer] /= neuron_nums
    return susp_distribution


def _compute_suspiciousness_scores(hit_spectrums: Dict[int, List[HitSpectrum]], approach='tarantula'):
    """HitSpectrumsを疑惑値に変換する
    :param hit_spectrums
    :return suspiciousness_scores
    """
    approachs = {
        'tarantula': _tarantula,
        'ochiai': _ochiai,
    }
    calc_susp_score = approachs[approach]
    
    suspiciousness_scores = []
    for layer, spectrums in hit_spectrums.items():
        suspiciousness = [
            SuspiciousnessScore(layer, i, calc_susp_score(hs))
            for i, hs
            in enumerate(spectrums)
        ]
        suspiciousness_scores += suspiciousness
    return suspiciousness_scores


def _compute_forward_impact(
        model,
        layer_id: int,
        dataset: Dataset):
    """順伝播を計算する
    :param model
    :param layer_id: 対象層のインデックス
    :param dataset: 計算対象のデータセット
    :return forward_impacts
    """
    if layer_id < 1:
        raise Exception('Not found previous layer: {}'.format(layer_id))


    previous_layer = model.get_layer(index=layer_id-1)
    get_activations = K.function(
        [model.input],
        [previous_layer.output]
    )
    _activations = get_activations([dataset.data])
    activations = np.mean(_activations[0], axis=0)
    
    target_layer = model.get_layer(index=layer_id)
    target_kernel = target_layer.get_weights()[0]
    
    forward_impacts = []
    for i in range(target_kernel.shape[0]):
        for j in range(target_kernel.shape[1]):
            o_i = activations[i]
            w_ij = target_kernel[i][j]
            forward_impacts.append(
                (
                    WeightId(layer_id, i, j),
                    o_i * w_ij
                )
            )
    return forward_impacts
