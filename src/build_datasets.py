# coding: UTF-8

from libdeeppatcher.arachne import Arachne
from libdeeppatcher import utils
from libdeeppatcher.types import Dataset
from libdeeppatcher import prepare
from libdeeppatcher import report
from libdeeppatcher import metrics
from IPython.display import clear_output
from keras import backend as K
import pandas as pd
import numpy as np

def _get_patched_weight_name(model_name, matrix_id, iter_count):
  return f'./arachne_results/{model_name}/{model_name}_arachne_weights_{matrix_id[0]}to{matrix_id[1]}_iter_{iter_count}.h5'


if __name__=='__main__':
  # モデル名とデータセット名のペアのリスト
  model_and_dataset = [
    ('mnist_dense', 'mnist_flat'),
    ('mnist_dense5', 'mnist_flat'),
    ('mnist_convolutional', 'mnist_conv'),
    ('mnist_conv4', 'mnist_conv'),
    ('fmnist_dense', 'fmnist_flat'),
    ('fmnist_dense5', 'fmnist_flat'),
    ('fmnist_convolutional', 'fmnist_conv'),
    ('fmnist_conv4', 'fmnist_conv'),
    ('resnet20', 'cifar10')
  ]

  for model_name, dataset_name in model_and_dataset:
    print(f'\nDATASET: {dataset_name}, MODEL: {model_name}\n')
    # 学習済みモデルをロード
    origin_model = prepare.load_model(model_name)
    tmp_model = utils.build_model(origin_model) # 修正後の重みをロードするための型となるモデル
    # データセットを分割（訓練・テストだけでなく，訓練データをさらに分割してる）
    _dataset = prepare.load_dataset(dataset_name)
    train_dataset, val_dataset, adv_dataset, test_dataset = _dataset
    # valの中から正しく予測できたデータだけ取り出す
    pos_dataset = utils.get_positive_dataset(origin_model, val_dataset)
    # advの中で間違いの多いtop5を取得
    repair_target_datasets = prepare.prepare_repair_target_datasets(origin_model, adv_dataset, n=5)

    # 正しく予測できたデータに対してメトリクスを計算(side-effect用)
    pos_res = metrics.calc_metrics(origin_model, pos_dataset)
    pos_res = pd.DataFrame.from_dict(pos_res, orient='index')
    # 副作用が起きた回数の列を追加(値域は[0,5])
    for topn in range(5):
      pos_res[f'broken_{topn+1}'] = 0 
    results = []
    # top1-5の各faultについて繰り返す
    for topn, (matrix_id, neg_dataset) in enumerate(repair_target_datasets):
      # 予測できなかったデータに対してメトリクスを計算(repairability用)
      neg_res = metrics.calc_metrics(origin_model, neg_dataset)
      neg_res = pd.DataFrame.from_dict(neg_res, orient='index')
      # 修正成功回数の列を追加(値域は[0,5])
      neg_res['success_num'] = 0
      for i in range(5):
        # 修正後の重みをロード
        patched_weights_name = _get_patched_weight_name(model_name, matrix_id, i)
        tmp_model.load_weights(patched_weights_name)
        # 修正できたデータのインデックス
        repaired_indeces = report.get_NG_to_OK_index(origin_model, tmp_model, neg_dataset)
        # 修正成功回数のカウント
        neg_res.loc[repaired_indeces, 'success_num'] += 1
        # 副作用がおきたデータのインデックス
        side_effected_indeces = report.get_OK_to_NG_index(origin_model, tmp_model, pos_dataset)
        # 副作用回数のカウント (列名broken_{n}はtop{n}の修正で副作用が何回起きたか,値域は[0,5])
        pos_res.loc[side_effected_indeces, f'broken_{topn+1}'] += 1
      results.append(neg_res)
    # faultごとのneg_dataに関するdfをひとつにまとめる
    _df_total = pd.concat(results)
    df_total = _df_total.reset_index(drop=True)
    dropped_cols = ['predicted_cls', 'label', 'label_cls', 'data']
    df_total.drop(dropped_cols, axis=1).to_csv(f'./repairability_dataset/{model_name}.csv')
    pos_res.drop(dropped_cols, axis=1).to_csv(f'./side_effect_dataset/{model_name}.csv')