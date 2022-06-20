from typing import NamedTuple
from typing import Any, List
import numpy as np


class Dataset(NamedTuple):
    data: Any
    label: Any


class WeightId(NamedTuple):
    layer_index: int
    i: int
    j: int


class Gradient(NamedTuple):
    weight_id: WeightId 
    grad_loss: np.float32


class Weight(NamedTuple):
    weight_id: WeightId
    val: np.float32


class MatrixId(NamedTuple):
    expected_label: int
    predicted_label: int


class PatchParam(NamedTuple):
    layer_indices: List[int]
    fire_threshold: float
    patch_range_rate: float
    decrement_rate: float


class HitSpectrum(NamedTuple):
    activate_pos: int
    not_activate_pos: int
    activate_neg: int
    not_activate_neg: int

        
class SuspiciousnessScore(NamedTuple):
    layer_index: int
    neuron_index: int
    suspiciousness: np.float
