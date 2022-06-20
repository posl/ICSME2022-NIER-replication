# coding: UTF-8

from libdeeppatcher.arachne import Arachne
from libdeeppatcher import utils
from libdeeppatcher.types import Dataset
from libdeeppatcher import prepare
from IPython.display import clear_output
from keras import backend as K
import os

if __name__=='__main__':
  model_and_dataset = [
    ('mnist_dense', 'mnist_flat'),
    ('mnist_dense5', 'mnist_flat'),
    ('mnist_convolutional', 'mnist_conv'),
    ('mnist_conv4', 'mnist_conv'),
    ('fmnist_dense', 'fmnist_flat'),
    ('fmnist_dense5', 'fmnist_flat'),
    ('fmnist_convolutional', 'fmnist_conv'),
    ('fmnist_conv4', 'fmnist_conv'),
  ]

  for model_name, dataset_name in model_and_dataset:
    print(f'\nDATASET: {dataset_name}, MODEL: {model_name}\n')
    origin_model = prepare.load_model(model_name)
    _dataset = prepare.load_dataset(dataset_name)
    train_dataset, val_dataset, adv_dataset, test_dataset = _dataset
    pos_dataset = utils.get_positive_dataset(origin_model, val_dataset)
    repair_target_datasets = prepare.prepare_repair_target_datasets(origin_model, adv_dataset, n=5)
    layer_indices = utils.get_repairable_dense_layer_indices(origin_model)[-1:]
    for topn, (matrix_id, neg_dataset) in enumerate(repair_target_datasets):
      print(f'\ntop{topn+1} fault: {matrix_id.expected_label} => {matrix_id.predicted_label}')
      for layer_index in layer_indices:
        if layer_index == 0:
          continue
        origin_model = prepare.load_model(model_name)
        arachne = Arachne(origin_model)
        arachne.repair_target_layer_index = layer_index
        print(f'target layer index: {layer_index}')
        # 重みの特定
        print('========= LOCALIZATION PHASE =========')
        localized = arachne.localize(neg_dataset)
        print('========= OPTIMIZATION PHASE =========')
        for i in range(5):
          print(f'rep: {i}')
          K.clear_session()
          repair_model = arachne.optimize(localized, neg_dataset, pos_dataset)
          save_name = model_name + '_arachne_weights_' + \
                      str(matrix_id[0]) + 'to' + str(matrix_id[1]) + \
                      '_iter_' + str(i)
          os.makedirs(f'./arachne_results/{model_name}/', exist_ok=True)
          repair_model.save_weights(f'./arachne_results/{model_name}/{save_name}.h5')