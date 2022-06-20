import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize
# import PyQt5 # Is it just a garbage ?
from .types import Dataset


def _plt_show():
    plt.tight_layout()
    plt.show()


def plot_result_after_fix(norm_result, adv_result, percentages, fix_rates, digit):
    """指定したdigitの修正後の認識率を各閾値、修正率ごとにplotする。
    青は元のデータ、赤はAdversaril example、
    緑の点線は、digit以外の数字の元のデータ
    灰色の点線は、digit以外のAdversarial Example
    :param original_results (ndarray): 元データの各閾値、修正率ごとの認識率
    :param adv_results (ndarray): AEの各閾値、修正率ごとの認識率
    :param thresholds (float list): 閾値のリスト
    :param fix_rates (float list): 修正率のリスト
    """
    np_fix_rate = np.array(fix_rates)
    np_original_results = np.array(norm_result)
    np_adv_results = np.array(adv_result)

    x = np_fix_rate
    for i in range(len(percentages)):
        plt.figure(facecolor="w")
        for j in range(10):
            if j != digit:
                y_o = np_original_results[i][:,j]
                y_a = np_adv_results[i][:,j]
                plt.plot(x, y_o, linestyle="dashed", linewidth=0.5, color="green")
                plt.plot(x, y_a, linestyle="dashed", linewidth = 0.5, color="gray")
        y_original = np_original_results[i][:, digit]
        y_adv = np_adv_results[i][:, digit]
        plt.plot(x, y_original,color="blue")
        plt.plot(x, y_adv, color="red")
        plt.ylim(-0.05, 1.05)
        plt.title("percentage: {}".format(percentages[i]))
        _plt_show()


def plot_full_connected_layer_outputs(outputs):
    """モデルの全結合層の出力をグラフとして表示する．
    :param outputs (ndarray list): モデルの全結合層の出力
    """
    plot_and_show_output(outputs)


def plot_conv_layer_outputs(outputs):
    """モデルの畳み込み層の出力をグラフとして表示する．
    :param outputs (ndarray list): モデルの畳み込み層の出力
    """
    plot_and_show_conv_outputs(outputs)


def plot_and_show_output(output, normalize=False, abs=False):
    """モデルの全結合の出力をグラフとして表示する．
    !! replace 'plot_full_connected_layer_outputs' !!
    :param output (ndarray list): 全結合層の各層のニューロンの出力値
    :param normalize (bool): 値を0~1正規化するかどうか
    :param abs (bool): 値を絶対値に変換するかどうか
    """
    if abs == True:
        output = np.abs(output)
    # 0-1正規化．
    if normalize == True:
        vmin = output.min()
        vmax = output.max()
        output = (output - vmin).astype(float) / (vmax - vmin).astype(float)
    idx = np.arange(0,output.shape[0])
    plt.figure(facecolor="w", figsize=(7,5))
    plt.grid(color="gray")
    plt.xticks(np.arange(0, 10, 1.0))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim([0,1])
    plt.bar(idx, output)
    _plt_show()


def plot_and_show_conv_outputs(outputs):
    """cnnの畳み込み層におけるfilterを画像として出力
    !! replace plot_conv_layer_outputs !!
    :param
    """
    filters = outputs.shape[2]
    plt.figure(facecolor="w", figsize=(20,10))
    vmax = outputs.max()
    vmin = outputs.min()
    print("vmax: {}   vmin: {}".format(vmax, vmin))
    color_bound = max(abs(vmax), abs(vmin))
    for i in range(filters):
        plt.subplot(filters/10 + 1, 10, i+1)
        plt.xticks([])
        plt.yticks([],[])
        plt.xlabel(f'output {i}')
        plt.imshow(outputs[:,:,i], cmap=plt.cm.coolwarm, vmin=(-color_bound), vmax=color_bound)
    _plt_show()


def display_imgs(x):
    """画像データを表示する
    :param x (ndarray): 
        画像データ
        ex: (1000, 28, 28, 1) => 1000件のmnistデータ
            (100, 32, 32, 3)  => 100件のcifar10データ
    """
    num_img = x.shape[0]
    img_shape = (28, 28) if x.shape[1] == 28 else (32, 32, 3)
    # 画像の表示
    fig = plt.figure( facecolor="w", figsize = ( 10, (num_img / 10 ) + 1 ) )
    idx = 1 # 画像表示位置設定のために必要
    for i in range( num_img ):
        ax = fig.add_subplot( ( num_img / 10 ) + 1 , 10, idx, xticks = [], yticks = [] )
        ax.imshow( x[i].reshape( img_shape ), cmap='gray' )
        idx += 1
    _plt_show()


def display_img_mnist(x):
    """mnistデータを画像で表示
    !! replace to 'display_imgs' !!
    :param x (ndarray): 28*28の画像データ
    """
    num_img = x.shape[0]
    idx = 1 # 画像をplotする位置設定のために必要
    # 画像の表示
    fig = plt.figure( facecolor="w", figsize = ( 10, (num_img / 10 ) + 1 ) )
    for i in range( num_img ):
        ax = fig.add_subplot( ( num_img / 10 ) + 1 , 10, idx, xticks = [], yticks = [] )
        ax.imshow( x[i].reshape( 28, 28 ), cmap='gray' )
        idx += 1
    _plt_show()


def display_img_mnist_digit(x, y, digit):
    """digitで指定したクラスの画像を表示する
    :param x: data
    :param y: class label of y
    :param digit: class (0 ~ 9)
    """
    index = _get_index_of_cls(y, digit)
    display_img_mnist(x[index])


def display_img_cifar10(x):
    """cifar10データを画像で表示する
    !! replace to 'display_imgs' !!
    :param x (ndarray): cifar10の画像データ
    """
    num_img = x.shape[0]
    idx = 1 # 画像をplotする位置設定のために必要
    # 画像の表示
    fig = plt.figure(facecolor="w", figsize=(10, (num_img / 10 ) + 1))
    for i in range(num_img):
        ax = fig.add_subplot((num_img / 10) + 1 , 10, idx, xticks=[], yticks=[])
        ax.imshow(x[i].reshape(32, 32, 3), cmap='gray')
        idx += 1
    _plt_show()


def _print_cmx(y_true, y_pred):
    """display confusion matrix
    :param y_true (ndarray):
        テストデータのラベル。one-hot表現の場合は元に戻しておくこと。
        ex. true_classes = np.argmax(y_test, axis=1)
    :param y_pred (ndarray): 予測結果

    how to use
    ------------------------------------------------
    predict_classes = model.predict_classes(x_test)
    true_classes = np.argmax(y_test, axis=1)
    _print_cmx(true_classes, predict_classes)
    ------------------------------------------------
    """
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    fig, ax = plt.subplots(facecolor="w", figsize=(13,8)) # (12,7) -> (13, 8)
    sn.set(font_scale=1.3)
    sn.heatmap(df_cmx, annot=True, fmt='g' ,square=True, cmap="GnBu")
    # 図が見切れるのを防ぐ（matplotlibのversionによっては発生しない）
    # https://qiita.com/yoshi65/items/90532732bf9d3875bec7
    ax.set_ylim(len(df_cmx), 0) 
    _plt_show()


def calc_and_show_cmx(model, dataset: Dataset):
    """Show confusion matrix.
    :param model: keras model
    :param dataset: data and label
    """
    predict_classes = np.argmax(model.predict(dataset.data), axis=1)
    true_classes = np.argmax(dataset.label, axis=1)
    _print_cmx(true_classes, predict_classes)


def plot_history(history):
    """plot training process.
    """
    # 精度の履歴をプロット
    plt.figure(facecolor="w", figsize = (12,7)) # (12,7)
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.figure(facecolor="w", figsize = (12,7)) # (12,7)
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


def plot_value_predictions(predictions, true_labels):
    """display model predictions on bar.
    predicted label is red, true label is blue.
    :param predictions: output of model.predict(x)
    :param true_labels: true labels of x
    """
    for pred, tlabel in zip(predictions, true_labels):
        plt.grid(False)
        plt.xticks([i for i in range(0,10,1)])
        plt.yticks([i/100 for i in range(0,101,25)])
        thisplot = plt.bar(range(10), pred, color="#777777")
        plt.ylim([0,1])
        predicted = np.argmax(pred)
        thisplot[predicted].set_color('red')
        thisplot[tlabel.argmax()].set_color('blue')
        _plt_show()


def plot_forward_propagation(preds):
    """forward propagation時の各出力層を可視化する。
    model.predict()の帰り値をそのまま渡すことはできないので注意。
    :param preds:
        return value of 
            - measure.get_delta_value
            - measure.calc_average_of_prediction
    """
    for pred in preds:
        dim = pred.ndim
        if dim == 1:
            plot_full_connected_layer_outputs(pred)
        elif dim == 3:
            plot_conv_layer_outputs(pred)
        else:
            raise Exception("something wrong...")


def plot_patched_or_broken(result_train, result_test):
    """'report.count_patched_and_broken'の返り値をplotする
    :param result_train, result_test: return value of 'count_patched_and_broken'
    """
    patched_train, broken_train = result_train
    patched_test, broken_test = result_test

    broken_train = [-x for x in broken_train]
    broken_test = [-x for x in broken_test]

    patched_train = np.array(patched_train)
    patched_test = np.array(patched_test)
    broken_train = np.array(broken_train)
    broken_test = np.array(broken_test)

    width = 0.3
    stride = width/2

    plt.figure(figsize=(20,10), facecolor="w")
    plt.grid(color="gray")
    h_axis = [i for i in range(0, 10, 1)]
    plt.xticks(h_axis)

    x = np.array(h_axis)

    plt.bar(x-stride, patched_train, width=width, color="royalblue", label="Patched(train)")
    plt.bar(x+stride, patched_test,  width=width, color="green",     label="Pathced(test)")
    plt.bar(x-stride, broken_train,  width=width, color="red",       label="Broken(train)")
    plt.bar(x+stride, broken_test,   width=width, color="orangered", label="Broken(test)")

    plt.legend(loc="upper right")

    _plt_show()
