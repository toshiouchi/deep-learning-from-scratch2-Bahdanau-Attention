# coding: utf-8
import sys
import os
sys.path.append('..')
sys.path.append('../dataset')
import numpy as np
import matplotlib.pyplot as plt
#from dataset import sequence
import sequence
from common.optimizer import Adam
from common.optimizer import SGD
from common.trainer import Trainer
from common.util import eval_seq2seq
from attention_seq2seq import AttentionSeq2seq
from attention_seq2seq import AttentionSeq2seq2
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq


# データの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

#print( "in train.py: id_to_char[14]:{}".format( id_to_char[14] ) )

# 入力文を反転
#x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2seq2(vocab_size, wordvec_size, hidden_size)
#model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam(lr=1e-3)
#optimizer = SGD(lr=1e-1)
trainer = Trainer(model, optimizer)

if os.path.isfile('AttentionSeq2seq2.pkl'):
    model.load_params()

acc_list = []
for epoch in range(max_epoch):
    #print( " train.py shape of t_train:{}".format( t_train.shape ))
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    #print( "in train.py, len(x_test):{}".format(len( x_test) ) )
    for i in range(len(x_test)):
    #for i in range(300):
        #print( "in train.py, i:{}".format( i ))
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse=False)

    acc = float(correct_num) / len(x_test)
    #acc = float(correct_num) / 300
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))


model.save_params()

# グラフの描画
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.show()
