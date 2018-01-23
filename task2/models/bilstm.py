from theano import tensor as T
import theano
import numpy as np

from utils import *

class BiLSTM(object):
    ''' Experimental BiLSTM with max-over-time pooling
    '''
    def __init__(self, emb, nh=256, nc=2, de=100, p_lstm_drop=0.5, p_drop=0.5,
            lr=0.001, train_emb=True, penalty=0., clip=None, lr_decay=0.):
        ''' Init Experimental BiLSTM model.

            Args:
            emb: Word embeddings matrix (num_words x word_dimension)
            nc: Number of classes
            de: Dimensionality of word embeddings
            p_lstm_drop: LSTM Dropout probability
            p_drop: Dropout probability
            lr: Initial learning rate
            train_emb: Boolean if the embeddings should be trained
            penalty: l2 regularization param
            clip: Gradient Clipping parameter (None == don't clip)
            decay: Learning rate decay parameter
        '''
        def recurrence(xi, mask, h_tm1, c_tm1, W_i, U_i, b_i, W_c, U_c, b_c, W_f, U_f, b_f, W_o2, U_o, b_o2, mask_in, mask_rec):
            x = xi*T.neq(mask, 0).dimshuffle(0,'x')
            x = dropout_scan(x, mask_in, dropout_switch, p_lstm_drop)

            x_i = T.dot(x, W_i) + b_i
            x_i = x_i*T.neq(mask, 0).dimshuffle(0,'x')

            x_f = T.dot(x, W_f) + b_f
            x_f = x_f*T.neq(mask, 0).dimshuffle(0,'x')

            x_c = T.dot(x, W_c) + b_c
            x_c = x_c*T.neq(mask, 0).dimshuffle(0,'x')

            x_o = T.dot(x, W_o2) + b_o2
            x_o = x_o*T.neq(mask, 0).dimshuffle(0,'x')

            h_tm1 = h_tm1*T.neq(mask, 0).dimshuffle(0,'x')
            h_tm1 = dropout_scan(h_tm1, mask_rec, dropout_switch, p_lstm_drop)

            i = hard_sigmoid(x_i + T.dot(h_tm1, U_i))
            f = hard_sigmoid(x_f + T.dot(h_tm1, U_f))
            c = f * c_tm1 + i * T.tanh(x_c + T.dot(h_tm1, U_c))
            o = hard_sigmoid(x_o + T.dot(h_tm1, U_o))
            h = o * T.tanh(c)
            return [h, c]

        # Embeddings
        self.emb = theano.shared(name='Words',
            value=emb.astype('float32'))

        # Source Output Weights
        self.w_o = theano.shared(name='w_o',
                value=he_normal((nh+nh,nc))
                .astype('float32'))
        self.b_o = theano.shared(name='b_o',
            value=np.zeros((nc,)).astype('float32'))

        # Update these parameters
        self.params = [self.emb, self.w_o, self.b_o]

        idxs = T.matrix()
        Y = T.imatrix()

        # get word embeddings based on indicies
        x_word = self.emb[T.cast(idxs, 'int32')]
        mask = T.neq(idxs, 0)*1
        x_word = x_word*mask.dimshuffle(0, 1, 'x')

        fwd_params, bck_params = bilstm_weights(de, nh)

        self.params += fwd_params + bck_params

        self.h0 = theano.shared(name='h0',
                                value=np.zeros((nh,),
                                dtype="float32"))

        dropout_switch = T.scalar()

        maskd1 = srng.binomial((x_word.shape[0], x_word.shape[-1]), p=1-p_lstm_drop, dtype='float32')
        maskd2 = srng.binomial((x_word.shape[0],nh), p=1-p_lstm_drop, dtype='float32')
        [h_fwd, cm], u = theano.scan(fn=recurrence,
                                sequences=[x_word.dimshuffle(1,0,2), idxs.dimshuffle(1,0)],
                                non_sequences=fwd_params+[maskd1, maskd2],
                                outputs_info=[T.alloc(self.h0, x_word.shape[0],nh), T.alloc(self.h0, x_word.shape[0],nh)],
                                n_steps=x_word.shape[1],
                                strict=True)

        maskd3 = srng.binomial((x_word.shape[0],x_word.shape[-1]), p=1-p_lstm_drop, dtype='float32')
        maskd4 = srng.binomial((x_word.shape[0],nh), p=1-p_lstm_drop, dtype='float32')
        [h_bck, c], u = theano.scan(fn=recurrence,
                                sequences=[x_word.dimshuffle(1,0,2)[::-1,:,:], idxs.dimshuffle(1,0)[::-1,:]],
                                non_sequences=bck_params+[maskd3, maskd4],
                                outputs_info=[T.alloc(self.h0, x_word.shape[0],nh), T.alloc(self.h0, x_word.shape[0],nh)],
                                n_steps=x_word.shape[1],
                                strict=True)

        h_bck = h_bck[::-1,:,:].dimshuffle(1,0,2)
        h_fwd = h_fwd.dimshuffle(1,0,2)
        h = T.concatenate([h_fwd, h_bck], axis=2)
        h = h.max(axis=1)
        h = dropout(h, dropout_switch, p_drop)

        pyx = T.nnet.nnet.softmax(T.dot(h, self.w_o) + self.b_o.dimshuffle('x', 0))
        pyx = T.clip(pyx, 1e-5, 1-1e-5)

        if train_emb:
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params])
            updates = Adam(L, self.params, lr2=lr, decay=lr_decay)
        else:
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params])
            updates = Adam(L, self.params[1:], lr2=lr, decay=lr_decay)

        self.train_batch = theano.function([idxs, Y, dropout_switch],
             [L, pyx.argmax(axis=1)], updates=updates, allow_input_downcast=True, on_unused_input='ignore')

        self.predict_proba = theano.function([idxs, dropout_switch],\
                pyx, allow_input_downcast=True, on_unused_input='ignore')
        self.predict = theano.function([idxs, dropout_switch],\
                pyx.argmax(axis=1), allow_input_downcast=True, on_unused_input='ignore')
        self.predict_loss = theano.function([idxs, Y, dropout_switch], [pyx.argmax(axis=1), L], allow_input_downcast=True)

    def __getstate__(self):
        return [x.get_value() for x in self.params]

    def __setstate__(self, weights):
        for x,w in zip(self.params, weights):
            x.set_value(w)

def bilstm_weights(de, nh):
    # Level 1 Bi-LSTM Weights
    Wf_i = theano.shared(name='w_i',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Uf_i = theano.shared(name='u_i',
                            value=he_normal((nh, nh))
                           .astype("float32"))
    bf_i = theano.shared(name='b_i',
                            value=np.zeros((nh), dtype="float32"))

    Wf_f = theano.shared(name='w_f',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Uf_f = theano.shared(name='u_f',
                            value=orthogonal_tmp((nh, nh))
                           .astype("float32"))
    bf_f = theano.shared(name='b_f',
                            value=np.ones((nh), dtype="float32"))

    Wf_c = theano.shared(name='w_c',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Uf_c = theano.shared(name='u_c',
                            value=orthogonal_tmp((nh, nh))
                           .astype("float32"))
    bf_c = theano.shared(name='b_c',
                            value=np.zeros((nh), dtype="float32"))

    Wf_o2 = theano.shared(name='woo',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Uf_o = theano.shared(name='uoo',
                            value=orthogonal_tmp((nh, nh))
                           .astype("float32"))
    bf_o2 = theano.shared(name='boo',
                            value=np.zeros((nh), dtype="float32"))
    #LSTM 2
    Wb_i = theano.shared(name='wb_i',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Ub_i = theano.shared(name='ub_i',
                            value=orthogonal_tmp((nh, nh))
                           .astype("float32"))
    bb_i = theano.shared(name='bb_i',
                            value=np.zeros((nh), dtype="float32"))

    Wb_f = theano.shared(name='wb_f',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Ub_f = theano.shared(name='ub_f',
                            value=orthogonal_tmp((nh, nh))
                           .astype("float32"))
    #ones
    bb_f = theano.shared(name='bb_f',
                            value=np.ones((nh), dtype="float32"))

    Wb_c = theano.shared(name='wb_c',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Ub_c = theano.shared(name='ub_c',
                            value=orthogonal_tmp((nh, nh))
                           .astype("float32"))
    bb_c = theano.shared(name='bb_c',
                            value=np.zeros((nh), dtype="float32"))

    Wb_o2 = theano.shared(name='wboo',
                            value=he_normal((de, nh))
                           .astype("float32"))
    Ub_o = theano.shared(name='uboo',
                            value=orthogonal_tmp((nh, nh))
                           .astype("float32"))
    bb_o2 = theano.shared(name='bboo',
                            value=np.zeros((nh), dtype="float32"))

    params_forward =  [Wb_i, Ub_i, bb_i,
                   Wb_c, Ub_c, bb_c,
                   Wb_f, Ub_f, bb_f,
                   Wb_o2, Ub_o, bb_o2]

    params_backward =  [Wf_i, Uf_i, bf_i,
                   Wf_c, Uf_c, bf_c,
                   Wf_f, Uf_f, bf_f,
                   Wf_o2, Uf_o, bf_o2]

    return params_forward, params_backward
