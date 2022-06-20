from vis.visualization import visualize_activation
from vis.utils import utils as vutils
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from keras import activations
from matplotlib import pyplot as plt


def _plt_show():
    plt.tight_layout()
    plt.show()

class VisModel:
    """keras-visを用いた可視化を行う
    """

    def __init__(self, model):
        self.vis_layer_idx = -1
        self.model = self._replace_softmax_with_linear(model)

    def _replace_softmax_with_linear(self, model):
        """出力層の活性化関数がsoftmaxの場合に可視化がうまくできないため、線形関数に取り替える
        詳細はvis-kerasのドキュメントを参照
        :param model: keras model
        :return model_with_linear: 出力層の活性化関数を線形関数に取り替えたモデル
        """
        model.layers[self.vis_layer_idx].activation = activations.linear
        model_with_linear = vutils.apply_modifications(model)
        return model_with_linear
        
    def visualize_activation_all_class(self):
        """すべてのクラスについて、ActivationMaximizationによる可視化を行う
        """
        filter_indices = [i for i in range(10)]
        for output_idx in filter_indices:
            img = visualize_activation(self.model, self.vis_layer_idx,
                                       filter_indices=output_idx, input_range=(0.,1.),
                                       input_modifiers=[Jitter(0.1)], max_iter=1000)
            plt.figure(facecolor='w')
            plt.axis('off')
            plt.imshow(img[..., 0], cmap="viridis")
            _plt_show()

    def visualize_activation(self, filter_indices):
        """指定したクラスについて、ActivationMaximizationによる可視化を行う
        :param filter_indices: 可視化対象のクラス
            ex: [0] => クラス0を最大化する入力値を可視化
                [1,2] => クラス1,2を同時に最大化する入力値を可視化
        """
        img = visualize_activation(self.model, self.vis_layer_idx, filter_indices=filter_indices,
                                   input_range=(0.1, 1.), input_modifiers=[Jitter(0.1)], max_iter=1000)
        plt.figure(facecolor='w')
        plt.axis('off')
        plt.imshow(img[..., 0], cmap="viridis")
        _plt_show()
