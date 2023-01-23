# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
#from dataset.mnist import load_mnist
sys.path.append('..')
sys.path.append('../dataset')
import sequence
#from two_layer_net import TwoLayerNet
from attention_seq2seq import AttentionSeq2seq
from attention_seq2seq import AttentionSeq2seq2


# データの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

print( "in train.py, shape of x_train:{}".format( x_train.shape ))
print( "in train.py, shape of t_train:{}".format( t_train.shape ))

# ハイパーパラメータの設定
batch_size = 20
vocab_size = len(char_to_id)
wordvec_size = 10
hidden_size = 10  # RNNの隠れ状態ベクトルの要素数
time_size = 35  # RNNを展開するサイズ
lr = 0.0001
max_epoch = 10
max_grad = 0.25

#network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = AttentionSeq2seq2(vocab_size, wordvec_size, hidden_size)

x_batch = x_train[:3]
t_batch = t_train[:3]

print( "in gradient_check_scratch2_ch08.py, shape of x_batch:{}".format( x_batch.shape ))
print( "in gradient_check_scratch2_ch08.py, shape of t_batch:{}".format( t_batch.shape ))

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#print( grad_numerical )
#print( grad_backprop )
#print( grad_backpropb )

#for key in grad_numerical.keys():
#    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
#    print(key + ":" + str(diff))

for i in range( len(grad_numerical )):
    diff = np.average( np.abs(grad_backprop[i] - grad_numerical[i]) )
    diff2 = np.sum( grad_backprop[i] - grad_numerical[i])  / np.sum( np.abs( grad_numerical[i] ) )
    #print( i + ":" + str(diff))
    #if i == 10 or i == 12:
    #    print( "i:{}, grad_numerical:{}".format( i, grad_numerical[i] ) )
    print( "grad_backprop[i]:{}".format( np.sum( np.abs(grad_backprop[i]) )) )
    print( "grad_numerical[i]:{}".format( np.sum( np.abs(grad_numerical[i]) )) )
    print( " i:{}, diff:{}, diff2:{}".format( i, diff, diff2 ) )