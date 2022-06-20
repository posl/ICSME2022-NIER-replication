from . import utils
from . import measure
from . import plot
from . import dpatchlib
from .types import Dataset
from typing import List, Optional

class DPatch:
    def __init__(self, model,
                 val_dataset: Dataset,
                 adv_dataset: Dataset,
                 repair_target_class: Optional[int]=None,
                 repair_target_layer: Optional[List[int]]=None):
        """
        :param model: repair taget_model
        :param val_dataset: validation data
        :param adv_dataset: repair taget data
        """
        self.model = model
        self.repaired_model = None

        self.val_dataset = val_dataset
        self.adv_dataset = adv_dataset

        self.target_val: Optional[Dataset] = None
        self.target_adv: Optional[Dataset] = None

        if repair_target_class is not None:
            self.repair_target_class = repair_target_class
            self.set_repair_target_class(repair_target_class)
        self.repair_target_layer = repair_target_layer

    def set_adv_data(self, adv_dataset: Dataset):
        self.adv_dataset = adv_dataset
        # 敵対的サンプルの種類が変更された際にtargetデータを再セットする。
        self._set_target_adv_data(self, self.repair_target_class)

    def set_repair_target_class(self, repair_target_class):
        self.repair_target_class = repair_target_class
        self._set_target_val_data(repair_target_class)
        self._set_target_adv_data(repair_target_class)
        print('[repair target_class]: {}'.format(repair_target_class))
        print('[mispredicted data]: {}/{}'
              .format(
                  self.target_adv.data.shape[0],
                  len(utils._get_index_of_cls(self.adv_dataset.label,
                                              repair_target_class))
              ))

    def _set_target_val_data(self, repair_target_class):
        index_of_target = utils._get_index_of_cls(self.val_dataset.label,
                                                  repair_target_class)
        self.target_val = Dataset(self.val_dataset.data[index_of_target],
                                  self.val_dataset.label[index_of_target])

    def _set_target_adv_data(self, repair_target_class):
        delta_index = utils.get_indices_of_mispred(self.model, self.adv_dataset)
        index_of_target = delta_index[repair_target_class]
        self.target_adv = Dataset(self.adv_dataset.data[index_of_target],
                                  self.adv_dataset.label[index_of_target])

    def set_repair_target_layer(self, repair_target_layer_index: List[int]):
        """修正対象層をセットする
        :param repair_target_layer_index: 修正対象層のインデックス
        """
        self.repair_target_layer = repair_target_layer_index

    def rank_repair_target_layer(self, fire_threshold=0, verbose=True):
        """修正可能な層のうち、誤認識に強く関わっていると考えられる層を順にリストアップする。
        :param fire_threshold:
        :param verbose: Trueの場合、層の順位付けに利用するメトリクスの情報を表示する
        """
        return measure.rank_repair_target_layer(
            self.model,
            utils.get_repairable_layer_indexs(self.model),
            self.target_val.data, self.target_adv.data,
            fire_threshold=fire_threshold,
            verbose=verbose)

    def visualize_model_repair_process(self, fire_threshold=0):
        if self.repair_target_layer is None:
            print("Please set repair target layer.('set_repair_target_layer()')")
            return None
        measure.summarize_repair_process(
            self.model,
            self.repair_target_layer,
            self.repair_target_class,
            self.val_dataset,
            self.adv_dataset,
            fire_threshold=fire_threshold)

    def repair(self, degration_limit=0.01, fire_threshold=0):
        self.repaired_model = dpatchlib.repair_model(
            self.model,
            self.repair_target_layer,
            self.repair_target_class,
            self.val_dataset,
            self.target_adv,
            degration_limit=degration_limit,
            fire_threshold=fire_threshold)

    def get_ranked_repair_taget_class(self):
        return measure.rank_repair_target_class(self.model,
                                                self.val_dataset,
                                                self.adv_dataset)