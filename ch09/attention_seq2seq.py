# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention
from ch08.attention_layer import BAttention
from common.base_model import BaseModel
from common.gradient import numerical_gradient


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:,-1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:,:,:H], dout[:,:,H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

class AttentionEncoder2():

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        #�G���R�[�_�[�Ŏg���p�����[�^�[���`�B
        embed_W = (rn(V, D) / 100).astype('f')
        gru_Wx = (rn(D, 3 * H) / np.sqrt(D)).astype('f')
        gru_Wh = (rn(H, 3 * H) / np.sqrt(H)).astype('f')
        gru_b = np.zeros(3 * H).astype('f')

        #���C���[���`
        self.embed = TimeEmbedding(embed_W)
        #self.gru = TimeGRU(gru_Wx, gru_Wh, gru_b, stateful=False)
        self.gru = TimeGRU(gru_Wx, gru_Wh, gru_b, stateful=True)

        #�p�����[�^�[�ƌ��z�̓��ꕨ���`�B
        self.params = self.embed.params + self.gru.params
        self.grads = self.embed.grads + self.gru.grads
        self.hs = None

        #���`��
    def forward(self, xs):
        h_initial = None
        self.gru.set_state(h_initial)
        xs = self.embed.forward(xs)
        hs = self.gru.forward(xs)
        return hs

        #�t�`��
    def backward(self, dhs):
        dout = self.gru.backward(dhs)
        dout = self.embed.backward(dout)
        
        self.grads[0][...] = self.embed.grads[0][...]
        self.grads[1][...] = self.gru.grads[0][...]
        self.grads[2][...] = self.gru.grads[1][...]
        self.grads[3][...] = self.gru.grads[2][...]        
        
        return dout

class AttentionDecoder2:
    def __init__(self, vocab_size, wordvec_size, hidden_size, embed_W,bAaffine1_W,bAaffine1_b,bAaffine2_W,bAaffine2_b,bAaffine3_W,bAaffine3_b,gru_Wx,gru_Wh,gru_b,affine_W,affine_b ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.V = V
        self.D = D
        self.H = H
        self.T = None

        #���C���[���`�B
        self.embed = TimeEmbedding(embed_W)
        self.battention = BAttention(bAaffine1_W, bAaffine1_b, bAaffine2_W, bAaffine2_b, bAaffine3_W, bAaffine3_b ) #Bahdanau attention �̃N���X�� attention_layers.py �Œ�`�B
        self.gru = TimeGRU(gru_Wx, gru_Wh, gru_b, stateful=False )
        #self.gru = TimeGRU(gru_Wx, gru_Wh, gru_b, stateful=True )
        self.affine = Affine(affine_W, affine_b)
        layers = [self.embed,  self.battention, self.gru, self.affine]

        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        #tensorflow https://tensorflow.classcat.com/2019/04/07/tf20-alpha-tutorials-sequences-nmt-with-attention/ �� Decoder ���Q�l�ɏ��`���B
    def forward(self, xs, dec_hs, enc_hs ):
        N, H = dec_hs.shape

        #h_initial = None
        #h_initial = enc_hs[:,-1,:]
        #print( "in attention_seq2seq AttentionDecoder2 def forward, shape of dec_hs:{}".format( dec_hs.shape ))
        #h_initial = dec_hs
        #print( "in attention_seq2seq AttentionDecoder2 def forward, shape of h_initial:{}".format( h_initial.shape ))
        self.N, self.T, self.H = enc_hs.shape

        #self.gru.set_state(h_initial)
        x = self.embed.forward(xs)
        c = self.battention.forward( dec_hs, enc_hs )
        c = np.expand_dims( c, axis = 1 )
        x = np.concatenate( [c, x], axis = -1 )
        output = self.gru.forward(x)
        dec_hs = np.squeeze( output, axis = 1 )
        output = np.reshape( output, (-1, output.shape[2]))
        score = self.affine.forward(output)

        return score, dec_hs

        #���`�����Q�l�ɋt�`���B
    def backward(self, dscore,  ddec_hs ):
        N,V = dscore.shape
        
        doutput = self.affine.backward(dscore)
        # output �� output �� dec_hs �ɕ��򂵂����Ƃ̋t�`���B sum�B
        doutput0 = np.expand_dims( ddec_hs, axis = 1 )
        doutput = np.expand_dims( doutput, axis = 1 ) + doutput0
        dx = self.gru.backward( doutput )
        #dh = self.gru.dh
        #print( "in attention_seq2seq AttentionDecoder2 def backward, shape of dh:{}".format( dh.shape ))
        # concatenate �̋t�`��
        dc = dx[:,0,0:self.H]
        dx = dx[:,:,self.H:]
        ddec_hs, denc_hs = self.battention.backward( dc )
        #denc_hs[:,-1,:] += dh
        #ddec_hs += dh

        dxs = self.embed.backward(dx)

        self.grads[0][...] = self.embed.grads[0][...]
        self.grads[1][...] = self.battention.grads[0][...]
        self.grads[2][...] = self.battention.grads[1][...]
        self.grads[3][...] = self.battention.grads[2][...]
        self.grads[4][...] = self.battention.grads[3][...]
        self.grads[5][...] = self.battention.grads[4][...]
        self.grads[6][...] = self.battention.grads[5][...]
        self.grads[7][...] = self.gru.grads[0][...]
        self.grads[8][...] = self.gru.grads[1][...]
        self.grads[9][...] = self.gru.grads[2][...]
        self.grads[10][...] = self.affine.grads[0][...]
        self.grads[11][...] = self.affine.grads[1][...]

        return denc_hs, ddec_hs, dxs


class AttentionSeq2seq2(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):

        rn = np.random.randn
        self.H = hidden_size
        self.V = vocab_size
        V, D, H = vocab_size, wordvec_size, hidden_size

        #�p�����[�^�[�̏����l��ݒ�B
        self.embed_W = (rn(V, D) / 100).astype('f')
        self.bAaffine1_W = (rn( (H), H ) / np.sqrt((H))).astype('f')
        self.bAaffine1_b = np.zeros( H ).astype('f')
        self.bAaffine2_W = (rn( H, H ) / np.sqrt(H)).astype('f')
        self.bAaffine2_b = np.zeros( H ).astype('f')
        self.bAaffine3_W = (rn( H, 1 ) / np.sqrt(H)).astype('f')
        self.bAaffine3_b = np.zeros( 1 ).astype('f')        
        self.gru_Wx = (rn((H+D), 3 * H) / np.sqrt((H+D))).astype('f')
        self.gru_Wh = (rn(H, 3 * H) / np.sqrt(H)).astype('f')
        self.gru_b = np.zeros(3 * H).astype('f')
        self.affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        self.affine_b = np.zeros(V).astype('f')

        #�R���X�g���N�g�̎��̈������X�g������Ă����B
        args1 = vocab_size, wordvec_size, hidden_size
        self.args2 = vocab_size, wordvec_size, hidden_size, self.embed_W,  self.bAaffine1_W, self.bAaffine1_b, self.bAaffine2_W, self.bAaffine2_b, self.bAaffine3_W,self.bAaffine3_b, self.gru_Wx, self.gru_Wh, self.gru_b, self.affine_W, self.affine_b        

        #�G���R�[�_�[���R���X�g���N�g�B�G���R�[�_�[��1�񂵂��R���X�g���N�g���Ȃ��̂ŁA������ OK�B
        self.encoder = AttentionEncoder2(*args1)

        # �p�����[�^�[�̓��ꕨ���`�B
        self.params = self.encoder.params + [ self.embed_W,self.bAaffine1_W, self.bAaffine1_b, self.bAaffine2_W, self.bAaffine2_b,self.bAaffine3_W,self.bAaffine3_b,self.gru_Wx,self.gru_Wh,self.gru_b, self.affine_W, self.affine_b]
        
        # ���z�̓��ꕨ���`�B
        self.d_grads = [ np.zeros_like(self.embed_W),np.zeros_like(self.bAaffine1_W), np.zeros_like(self.bAaffine1_b), np.zeros_like(self.bAaffine2_W), np.zeros_like(self.bAaffine2_b),np.zeros_like(self.bAaffine3_W),np.zeros_like(self.bAaffine3_b),np.zeros_like(self.gru_Wx),np.zeros_like(self.gru_Wh),np.zeros_like(self.gru_b), np.zeros_like(self.affine_W), np.zeros_like(self.affine_b)]
        self.grads =  self.encoder.grads +  self.d_grads

    def forward(self, xs, ts ):
        N, T = ts.shape
        Nxs, Txs = xs.shape
        self.N = N
        self.T = T
        self.Txs = Txs
        start_id = ts[0,0] # start_id ���� 14 �����A������`�B
        #����AttentionSeq2seq ���� loss �̌v�Z�� decoder_ts ���g���悤�ɂȂ��Ă��邪�Ahttps://tensorflow.classcat.com/2019/04/07/tf20-alpha-tutorials-sequences-nmt-with-attention/
        #�� train_step �ł́Afor ����range(1,T)�Ƃ��邱�Ƃɂ��A�ӎ��I��1���炵�Ă���̂ŁA�V���� decoder_ts2 ���`���Ă�����g���B
        decoder_xs, decoder_ts, decoder_ts2 = ts[:,:], ts[:, 1:], ts[:,:]

        h = self.encoder.forward(xs) #AttentionEncoder2 ���R�[���B
        #print( "in attention_seq2seq class AttentionSeq2seq2 def forward, h[:,:,:]:{}".format(  h[:,:,:] ))


        self.layers = [] # for ���[�v�̒��� Decoder2 ���C���[�𕡐��R���X�g���N�g����̂ŁA���̃��C���[���`�B
        self.layers2 = [] # for ���[�v�̒��� SoftMaxWithLoss ���C���[�𕡐��R���X�g���N�g����̂ŁA���̃��C���[���`�B

        dec_hs = np.sum( h, axis = 1 ) # �f�R�[�_�[�ɓ��͂���B���ԁBtensorflow �ł́A�G���R�[�_�[�̍ŏI��Ԃ����A�G���R�[�_�[�̉B���Ԃ� sum �̕����ǂ����ʂ��ł�悤���B
        dec_input = np.zeros( ( self.N, 1 ), dtype=np.int64  ) #�f�R�[�_�[�ɓ��͂���A���̓e���\���̊�������B
        dec_input[:,0] = start_id #�f�R�[�_�[�ɓ��͂���A���̓e���\���� t = 0 �́Astart_id�B
        total_loss = 0
        
        for t in range( 1, T  ): # tensorflow �̃y�[�W�ɂ���悤�ɁA1,T �Ń��[�v�B
            layer = AttentionDecoder2(*self.args2) # AttentionDecoder2 ���C���[���R���X�g���N�g
            layer2 = SoftmaxWithLoss() # SoftmaxWithLoss ���C���[���R���X�g���N�g
            predictions, dec_hs = layer.forward( dec_input, dec_hs, h ) # AttentionDecoder2 ���C���[���R�[��
            loss = layer2.forward( predictions, decoder_ts2[:,t] ) # SoftmaxWithLoss ���C���[���R�[������ loss ���v�Z�B
            total_loss += loss # total_loss ���v�Z�B
            self.layers.append(layer) 
            self.layers2.append( layer2 )
            dec_input[:,0] = decoder_xs[:, t] #�f�R�[�_�[�ɓ��͂�����̓e���\����ݒ�B

        return total_loss / ( T - 1 )

    def backward(self, dout=1):

        d_grads = [0,0,0,0,0,0,0,0,0,0,0,0] #�f�R�[�_�[�ɂ�����A�e�p�����[�^�[�̌��z���`�B
        ddec_hs = np.zeros( ( self.N, self.H ), dtype=np.float64 ) 
        dh_sum = np.zeros( ( self.N, self.Txs, self.H ), dtype=np.float64 )
        for t in reversed(range( 1, self.T ) ): # forward �� 1,T �̃��[�v���t�ɂ��ǂ�B
            t2 = t - 1 # layer �����ʂ��邽�߂̃p�����[�^�[ t2 ���쐬�B
            layer = self.layers[t2] # ���C���[���Ăяo���B
            layer2 = self.layers2[t2]
            dpredictions = layer2.backward( dout ) # SoftmaxWithLoss ���C���[�̋t�`��
            dh, ddec_hs, ddec_input = layer.backward(dpredictions, ddec_hs ) # AttentionDecoder2 �̋t�`��
            dh_sum += dh # h �̕���ɑ΂���t�`�� sum�B 
            for i, grad in enumerate(layer.grads): # AttentionDecoder2 ���C���[�̌��z�̕���ɂ�� sum
                d_grads[i] += grad

        dxs = self.encoder.backward(dh_sum) # AttentionEncoder �̋t�`��

        self.grads[0][...] = self.encoder.grads[0][...] # AttentionEncoder2 �̌��z���i�[
        self.grads[1][...] = self.encoder.grads[1][...]
        self.grads[2][...] = self.encoder.grads[2][...]
        self.grads[3][...] = self.encoder.grads[3][...]
        for i, d_grad in enumerate(d_grads): # AttentionDecoder2 �̌��z���i�[
            i2 = i + len( self.encoder.grads )
            self.grads[i2][...] = d_grad    
        
        return dxs

    def generate(self, xs, start_id, sample_size):
        N, T = xs.shape

        h = self.encoder.forward(xs)
        sampled = []

        self.layers = []

        dec_hs = np.sum( h, axis = 1 )
        dec_input = np.zeros( ( 1, 1 ), dtype=np.int64  )
        dec_input[0,0] = start_id
        sample_id = start_id

        for t in range( sample_size ):
            layer = AttentionDecoder2(*self.args2)
            predictions, dec_hs = layer.forward( dec_input, dec_hs, h )
            self.layers.append(layer)
            sample_id = np.argmax( predictions )
            sampled.append(sample_id)
            # �\�����ꂽ ID �����f���ɖ߂����
            dec_input[0,0] = sample_id
        
        return sampled
