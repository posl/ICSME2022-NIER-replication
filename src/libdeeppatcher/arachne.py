import random
import shutil
from pathlib import Path
import gc

import keras
from keras.models import load_model
import numpy as np
from keras import backend as K
from typing import NamedTuple, List, Optional
from .types import Dataset, Gradient, WeightId, Weight
from . import utils
from . import arachnelib
import time
import copy


def printfunc(func):
    def wrapper(*args, **kwargs):
        print('[{}] Start ===>'.format(func.__name__))
        result = func(*args, **kwargs)
        print('<=== End [{}]'.format(func.__name__))
        return result
    return wrapper


class Arachne:
    def __init__(self, model):
        self.num_particles = 100
        self.num_iterations = 100
        self.model = model
        self.origin_weights = model.get_weights()

        # 修正対象の選択
        # default値は最終層
        self.repair_target_layer_index = len(model.layers) - 1

    def reset_model(self):
        """モデルをコンパイルしてオリジナルの重みをセットする
        K.clear_session()を使用したあとはこれを用いる必要がある
        """
        self.model = utils.build_model(self.model)
        self.reset_weights()

    def reset_weights(self):
        """オリジナルの重みをセットしなおす
        """
        self.model.set_weights(self.origin_weights)


    def localize(self,
                 fault_dataset: Dataset,
                 num_grad: Optional[int] = None) -> List[WeightId]:
        """Localisation method of Arachne.
        _compute_gradient と _compute_forward_impact でグラフを生成するので、
        K.clear_session()でリセットしている。

        :param model: a model to repair
        :param fault_dataset: a set op inputs that reveal the fault
        :param num_grad:
            the number of neural weight candidates to choose based on gradient loss.
        :return ids of weight of repairing target
        """
        # Compute weight candidates by gradient loss.
        grad_candidates: List[Gradient] = \
            self._compute_gradient(self.model, fault_dataset)
        grad_candidates.sort(key=lambda grd: grd.grad_loss, reverse=True)

        _num_grad = num_grad if num_grad is not None else len(fault_dataset.data)*20
        _num_grad = min(_num_grad, len(grad_candidates))
        print('num_grad: {}'.format(_num_grad))

        pool = {}
        for i, (weight_id, grad_loss) in enumerate(grad_candidates[:_num_grad]):
            print(i, end=' ')
            fwd_imp = self._compute_forward_impact(self.model,
                                                   fault_dataset,
                                                   weight_id)
            pool[i] = (weight_id, grad_loss, fwd_imp)

        weight_ids: List[WeightId] = self._extract_pareto_front(pool)

        # Reset session
        K.clear_session()

        # Restore model
        self.reset_model()

        return weight_ids


    def _compute_gradient(self, model, dataset: Dataset) -> List[Gradient]:
        """Compute gradient loss.
        :param model:
        :param dataset: a negative data ??
        :return candidates: List of gradient loss with weight id.
        """
        loss_func = keras.losses.get(model.loss)
        # 最終層の一つ前の層のみを対象とする??
        layer_index = self.repair_target_layer_index
        layer = model.layers[layer_index]

        sess = K.get_session()
        loss = loss_func(dataset.label, model.output)

        get_grad_kernel = K.gradients(loss, layer.kernel)[0]
        grad_kernel = sess.run(get_grad_kernel,
                               feed_dict={model.input: dataset.data})

        get_grad_output = K.gradients(loss, layer.output)[0]
        # 複数データの場合は平均値を取る。（1件の場合もこれでOK.）
        grad_output = sess.run(get_grad_output,
                               feed_dict={model.input: dataset.data})
        grad_output = np.mean(grad_output, axis=0)

        candidates: List[Gradient] = []
        # i, j はlayer n-1 の i 番目と layer n の j番目みたいな感じ
        for j in range(grad_kernel.shape[1]):
            dl_do = grad_output[j]
            for i in range(grad_kernel.shape[0]):
                do_dw = grad_kernel[i][j]
                grad = Gradient(weight_id=WeightId(layer_index, i, j),
                                grad_loss=np.abs(dl_do * do_dw))
                candidates.append(grad)
        return candidates


    def _compute_forward_impact(
            self,
            model,
            fault_dataset: Dataset,
            weight_id: WeightId):
        """Compute forward impact.
        前のレイヤーからの出力値 * kernelの重みパラメータをかけ合わせてfoward impactを計算
        :param model:
        :param fault_dataset:
        :param weight_id:
        :return value of forward propagation
        """
        layer_index, neural_weight_i, neural_weight_j = weight_id
        if layer_index < 1:
            raise Exception('Not found previous lyaer: {}'.format(layer_index))

        # Calc output value of layer 'i'
        previous_layer = model.get_layer(index=layer_index-1)
        get_activations = K.function([model.input, K.learning_phase()],
                                     [previous_layer.output])

        # 下記はmnist_denseであれば問題なく動いたが、
        # cifar10で実行したところエラーになった。
        # そもそもfault_datasetを渡しているが、fault_dataset.dataのほうが正しそう？
        # なんで今まで動いていたの？
        # activations = get_activations(fault_dataset)
        activations = get_activations([fault_dataset.data])

        # 一件のデータであれば下記でOK
        # o_i = activations[0][0][neural_weight_i] # TODO correct??
        # 複数データの場合は平均を取る(1件データにも対応)
        activations = np.mean(activations[0], axis=0)
        o_i = activations[neural_weight_i]

        # Get weight value of layer 'j'
        target_layer = model.get_layer(index=layer_index)
        # sessionを使わなくても直接get_weights()で値を取れる。[0] => kernel, [1] => bias
        w_ij = target_layer.get_weights()[0][neural_weight_i][neural_weight_j]

        return np.abs(o_i * w_ij)


    def _extract_pareto_front(self, pool) -> List[WeightId]:
        """Extract pareto front weight ids from candidates.
        :param pool: { weight_id, grad_loss, fwd_imp }
        :return weight_ids_of_pareto_front: pareto front weight ids
        """
        # grad_lossとfwd_impのpareto_frontを抽出。
        scores = np.array([[grad_loss, fwd_imp]
                           for weight_id, grad_loss, fwd_imp
                           in pool.values()])
        pareto_front_indices: List[int] = arachnelib.identify_pareto(scores)
        pareto_front = scores[pareto_front_indices] # np.ndarray([grad_loss, fwd_imp])

        weight_ids_of_pareto_front: List[WeightId] = []
        for weight_id, grad_loss, fwd_imp in pool.values():
            for _grad_loss, _fwd_imp in pareto_front:
                if grad_loss == _grad_loss and fwd_imp == _fwd_imp:
                    weight_ids_of_pareto_front.append(weight_id)
                    break
        return weight_ids_of_pareto_front


    def optimize(self,
                 weight_ids: List[Weight],
                 fault_dataset: Dataset,
                 pos_dataset: Dataset):
        """
        """
        self.reset_model()
        # Initialize particle positions
        locations: List[List[Weight]] = \
            self._get_initial_particle_positions(weight_ids, self.num_particles)

        # The initial velocity of each particle is set to zero
        velocities = np.zeros((self.num_particles, len(weight_ids)))

        # Compute velocity bounds
        velocity_bounds = self._get_velocity_bounds(self.model)

        # Sample 200 positive inputs
        sample_pos_dataset = \
            utils.get_dataset_by_random_sampling(pos_dataset, num=200)

        # Initialize for PSO search
        personal_best_positions = copy.deepcopy(locations)

        personal_best_scores = \
            self._initialize_personal_best_scores(locations,
                                                  sample_pos_dataset,
                                                  fault_dataset)

        best_particle = np.argmax(personal_best_scores) # Larger is better

        global_best_position = personal_best_positions[best_particle]

        # ここまで初期化処理なので、別の場所に持っていきたいne

        print('[INITIAL STATE]')
        print('===================')
        print('[prediction score(sample pos)]: ',
              self.model.evaluate(*sample_pos_dataset, verbose=0))
        print('[prediction score(neg)]: ', self.model.evaluate(*fault_dataset, verbose=0))
        print('[best score]: {}'.format(max(personal_best_scores)))
        print('===================')
        print()

        # Search
        print('[Search Start]')
        history = []
        count = 0
        max_score = 0
        # PSO uses ... the maximum number of iterations 100
        for t in range(self.num_iterations):
            print('iter: ', t)
            start_time = time.time()
            # PSO uses a population size of 100
            for n in range(self.num_particles):
                print('*', end='')
                x = []
                layer_indices = []
                nw_i = []
                nw_j = []
                for weight_id, val in locations[n]:
                    x.append(val)
                    layer_indices.append(weight_id.layer_index)
                    nw_i.append(weight_id.i)
                    nw_j.append(weight_id.j)

                x = np.array(x)
                v = velocities[n]

                # Update velocity
                _get_weight_values = lambda weights: [w.val for w in weights]
                p = _get_weight_values(personal_best_positions[n])
                g = _get_weight_values(global_best_position)
                new_v = self._update_velocity(x,
                                              v,
                                              p,
                                              g,
                                              velocity_bounds,
                                              layer_indices)
                velocities[n] = new_v

                # Update position
                new_x = self._update_position(x, new_v)
                for i, n_x in enumerate(new_x):
                    _weight_id = WeightId(layer_indices[i],
                                          nw_i[i],
                                          nw_j[i],)
                    _weight_val = n_x
                    locations[n][i] = Weight(weight_id=_weight_id,
                                             val=_weight_val)

                # Update personal best
                self.reset_weights()
                self._mutate(self.model, locations[n])
                score = self._criterion(self.model,
                                        sample_pos_dataset,
                                        fault_dataset)
                if personal_best_scores[n] < score:
                    personal_best_scores[n] = score
                    # location[n]をそのまま代入してしまうと、リストの参照をコピーすることになるので注意
                    personal_best_positions[n] = locations[n][:]

            # Update global best (personal bestの中からベストなものを取得)
            best_particle = np.argmax(personal_best_scores)
            personal_best_score = max(personal_best_scores)
            global_best_position = personal_best_positions[best_particle]
            print()
            # scoreが高くなった場合はカウントを0にリセット
            if max_score < personal_best_score:
                print('[UPDATE BEST SCORE]')
                self.reset_weights()
                self._mutate(self.model, global_best_position)
                count = 0
                max_score = personal_best_score
                print('===================')
                print('[prediction score(sample pos)]: ', self.model.evaluate(*sample_pos_dataset, verbose=0))
                print('[prediction score(neg)]: ', self.model.evaluate(*fault_dataset, verbose=0))
                print('[best score]: {}'.format(max(personal_best_scores)))
                print('===================')
            # そうでなければcountを増やす。
            else:
                count += 1


            # Stop earlier if it fails to find a better patch
            # countが10になったら抜ける
            if count == 10:
                print('fail to find a better patch...')
                break

            # Add current best
            history.append(max(personal_best_scores))

            end_time = time.time()
            print('time: ', end_time - start_time)
            print()

        print('[FINAL RESULT]')
        print('[best score]: {}'.format(max(personal_best_scores)))

        repaired = utils.build_model(self.model)
        repaired.set_weights(self.origin_weights)
        self._mutate(repaired, global_best_position)
        return repaired


    def _mutate(self, model, locations):
        """Mutate model with location.
        modelを作り直す必要はある？？ => ない。ただし、呼び出す側でモデルをrestoreする必要がある？
        """
        for location in locations:
            (layer_index, nw_i, nw_j), val = location
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            weights[0][nw_i][nw_j] = val
            layer.set_weights(weights)


    def _initialize_personal_best_scores(
            self,
            locations,
            pos_dataset,
            fault_dataset):
        """
        """
        print('[Initialize personal best scores]')
        print('Start =>')
        personal_best_scores = []
        origin_weights = self.model.get_weights()

        for location in locations:
            print('*', end='')

            # Reset model weights
            self.model.set_weights(origin_weights)

            # Mutate model
            self._mutate(self.model, location)

            # Calc fitness
            fitness = self._criterion(self.model,
                                      pos_dataset,
                                      fault_dataset)

            personal_best_scores.append(fitness)

        print('<= End')
        # Restore model
        self.reset_weights()

        return personal_best_scores


    def _get_initial_particle_positions(
            self,
            weight_ids: List[WeightId],
            num_particles: int) -> List[List[Weight]]:
        """Get initial particle positions.
        論文2.4.2章を参照
        """
        # locations[i][j]
        locations = [[None] * len(weight_ids)
                     for i in range(num_particles)]

        for n_w, weight_id in enumerate(weight_ids):
            layer_index, nw_i, nw_j = weight_id
            sibling_weights = []

            # L_{n}
            layer = self.model.get_layer(index=layer_index)
            target_weights = layer.get_weights()[0]
            for j in range(target_weights.shape[1]):
                for i in range(target_weights.shape[0]):
                    sibling_weights.append(target_weights[i][j])

            # Each element of a particle vector
            # is sampled from a normal distribution
            # defined by the mean and the standard deviation
            # of all sibling neural weighs.
            mu = np.mean(sibling_weights)
            std = np.std(sibling_weights)
            samples = np.random.normal(loc=mu,
                                       scale=std,
                                       size=num_particles)

            for n_p in range(num_particles):
                sampled_val = samples[n_p]
                locations[n_p][n_w] = Weight(weight_id, sampled_val)

        return locations


    def _get_velocity_bounds(self, model):
        """Get velocity bounds.
        論文4.2章を参照

        "W is the set of all neural weights
        between our target layer and the preceding one."

        wb = np.max(all_weights) - np.min(all_weights)
        vb = (wb / 5, wb * 5)

        :param model:
        :return: dictionary whose key is layer index
                 and value is velocity bounds
        """
        # Range from 1: Use preceding layer
        # To #layers-1: Except for final output layer
        velocity_bounds = {}
        layer_index = self.repair_target_layer_index
        layer = model.get_layer(index=layer_index)

        # Get all weights
        all_weights = []
        target_weights = layer.get_weights()[0]
        for j in range(target_weights.shape[1]):
            for i in range(target_weights.shape[0]):
                all_weights.append(target_weights[i][j])

        # Velocity bounds defined at equations 5 and 6
        wb = np.max(all_weights) - np.min(all_weights)
        vb = (np.min(all_weights), np.max(all_weights))

        velocity_bounds[layer_index] = vb
        print('max: {}  min: {}'.format(np.max(all_weights), np.min(all_weights)))
        print(velocity_bounds)
        return velocity_bounds

    def _criterion(
            self,
            model, 
            pos_dataset: Dataset,
            fault_dataset: Dataset,
            batch_size=32):
        """
        """
        def _count_correct(dataset: Dataset):
            preds = model.predict(dataset.data, verbose=0).argmax(axis=1)
            expected = dataset.label.argmax(axis=1)
            return sum(preds == expected)

        # "N_{patched} is the number of inputs in I_{neg}
        # whose output is corrected by the current patch"
        n_patched = _count_correct(fault_dataset)

        # "N_{intact} is the number of inputs in I_{pos}
        # whose output is still correct"
        n_intact = _count_correct(pos_dataset)

        # The model loss values obtained from these inputs sets
        # Loss(I_{neg}) and Loss(I_{pos}) are used
        # to reflect improvement and damages
        # that do one to both numerators nad denominators
        # so that one can provide guidance
        # when the other becomes zero, and vice verse
        loss_input_neg = model.evaluate(*fault_dataset, verbose=0)[0]
        loss_input_pos = model.evaluate(*pos_dataset, verbose=0)[0]

        fitness = ((n_patched + 1) / (loss_input_neg + 1)) + \
                  ((n_intact + 1) / (loss_input_pos + 1))

        return fitness

    def _update_position(self, x, v):
        """Update position x
        x_{t+1} <- x_{t} + v_{t+1}
        :param x: x_{t}
        :param v: v_{t+1}
        :return x_{t+1}
        """
        return x + v

    def _update_velocity(self, x, v, p, g, vb, layer_index):
        """Update velocity.
        :param x: current position
        :param v: current velocity
        :param p: personal best position
        :param g: global best position
        :param vb: velocity bounds computed in each layer
        :param layer_index:
        :return: new velocity
        """
        # "We follow the general recommendation
        # in the literature and set both to 4.1"
        # 論文4.2章を参照（4.1という値が指定されているのはここ。）
        phi = 4.1
        # "Equation 3"
        chi = 2 / (phi - 2 + np.sqrt(phi * phi - 4 * phi))
        # "Equation 2"
        ro1 = random.uniform(0, phi)
        ro2 = random.uniform(0, phi)
        # TODO Using same value 'chi'
        #  to 'w', 'c1', and 'c2' in PSO hyper-parameters?
        new_v = chi * (v + ro1 * (p - x) + ro2 * (g - x))

        # layer_indexは今のところ全部同じ
        lower_bound, upper_bound = vb[layer_index[0]]
        new_v[new_v < lower_bound] = lower_bound
        new_v[new_v > upper_bound] = upper_bound

        return new_v
