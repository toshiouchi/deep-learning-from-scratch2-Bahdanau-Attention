# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Softmax
from common.time_layers import TimeAffine

class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1)#.repeat(T, axis=1)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H)#.repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:,t,:])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:,t,:] = dh

        return dhs_enc, dhs_dec
        
# tensorflow https://tensorflow.classcat.com/2019/04/07/tf20-alpha-tutorials-sequences-nmt-with-attention/ の class BahdanauAttention の score の計算。
class Score:
    def __init__(self, bAaffine1_W, bAaffine1_b, bAaffine2_W, bAaffine2_b, bAaffine3_W, bAaffine3_b ):
        self.affine1 = TimeAffine(bAaffine1_W, bAaffine1_b)
        self.affine2 = TimeAffine(bAaffine2_W, bAaffine2_b)
        self.affine3 = TimeAffine(bAaffine3_W, bAaffine3_b)
        layers = [self.affine1,self.affine2,self.affine3]
        
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads        

        #順伝搬
    def forward(self,  query, values ):
        N, T, H = values.shape
        N, H = query.shape
        self.T = T
        
        hidden_with_time_axis = np.expand_dims( query, axis = 1 )
        self.values = values

        score0 = self.affine1.forward( values )
        score1 = self.affine2.forward( hidden_with_time_axis )

        score2 = score0 + score1

        score3 = np.tanh( score2 )

        self.score3 = score3

        score = self.affine3.forward( score3 )

        return score

        #順伝搬を参考に逆伝搬
    def backward( self,  dscore ):

        dscore3 = self.affine3.backward( dscore )

        dscore2 = dscore3 * ( 1 - self.score3 ** 2 ) # tanh の逆伝搬。
        
        dscore0 = dscore2
        # dscore1 の broadcast の逆伝搬
        dscore1 = np.sum( dscore2, axis = 1 )
        dscore1 = np.expand_dims( dscore1, axis = 1 )

        dhidden_with_time_axis = self.affine2.backward( dscore1 )
        dquery = np.squeeze( dhidden_with_time_axis, axis = 1 )

        dvalues = self.affine1.backward( dscore0 )

        self.grads[0][...] = self.affine1.grads[0][...]
        self.grads[1][...] = self.affine1.grads[1][...]
        self.grads[2][...] = self.affine2.grads[0][...]
        self.grads[3][...] = self.affine2.grads[1][...]
        self.grads[4][...] = self.affine3.grads[0][...]
        self.grads[5][...] = self.affine3.grads[1][...] 
       
        return dquery, dvalues

# tensorflow https://tensorflow.classcat.com/2019/04/07/tf20-alpha-tutorials-sequences-nmt-with-attention/ の class BahdanauAttention。
class BAttention:
    def __init__(self, bAaffine1_W, bAaffine1_b, bAaffine2_W, bAaffine2_b, bAaffine3_W, bAaffine3_b ):
        self.score = Score(bAaffine1_W, bAaffine1_b, bAaffine2_W, bAaffine2_b, bAaffine3_W, bAaffine3_b)
        layers = [self.score]
        
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads        
        
        self.T = None
        self.Softmax_layer = Softmax()
        
        #順伝搬
    def forward(self,  query, values ):

        N, H = query.shape
        N, T, H = values.shape
        self.T = T
        
        self.values = values

        score = self.score.forward( query, values )
        
        attention_weights = self.Softmax_layer.forward( score )
        
        self.attention_weights = attention_weights #attention_weights は ( batch_size, T_max_length, 1 )
        
        context_vector = attention_weights * values # ( batch_size, T_max_length, hidden_size ) attemtop_weights はブロードキャスト
        context_vector = np.sum( context_vector, axis = 1 )

        return context_vector
       
        #順伝搬を参考に逆伝搬
    def backward(self, dcon ):
        N, H = dcon.shape
       
        dcon = np.expand_dims( dcon, axis = 1 )
        dcon = np.repeat( dcon, self.T, axis = 1 )

        dattention_weights = dcon * self.values
        # attention_weights の boradcast の逆伝搬
        dattention_weights = np.sum( dattention_weights, axis = 2 )
        dattention_weights = np.expand_dims( dattention_weights, axis = 2 )
        
        dvalues0 = dcon * self.attention_weights
       
        dscore = self.Softmax_layer.backward( dattention_weights )

        dquery, dvalues = self.score.backward( dscore )

        # values の分岐による sum。
        dvalues += dvalues0

        self.grads[0][...] = self.score.grads[0][...]
        self.grads[1][...] = self.score.grads[1][...]
        self.grads[2][...] = self.score.grads[2][...]
        self.grads[3][...] = self.score.grads[3][...]
        self.grads[4][...] = self.score.grads[4][...]
        self.grads[5][...] = self.score.grads[5][...]       
       
        return dquery, dvalues

