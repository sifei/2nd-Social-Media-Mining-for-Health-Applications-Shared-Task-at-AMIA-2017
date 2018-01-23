from theano import tensor as T
import theano
import numpy as np

from utils import *

class AttBoW(object):
    ''' Experimental Attenttive BoW model.
    '''
    def __init__(self, emb, nc=2, de=300, p_drop=0.5, nh=2048, train_emb=True,
            na=30, penalty_p=0, penalty=0, lr=0.001, decay=0., clip=None):
        ''' Init Experimental Attenttive BoW model.

            Args:
            emb: Word embeddings matrix (num_words x word_dimension)
            nc: Number of classes
            de: Dimensionality of word embeddings
            p_drop: Dropout probability
            nh: Hidden layer dimensions
            train_emb: Boolean if the embeddings should be trained
            na: Number of attention vectors
            penalty_p: attention regularization param
            penalty: l2 regularization param
            lr: Initial learning rate
            decay: Learning rate decay parameter
            clip: Gradient Clipping parameter (None == don't clip)
        '''
        self.emb = theano.shared(name='Words',
            value=np.asarray(emb, dtype='float32'))
        self.w_o = theano.shared(
                value=he_normal((nh, nc))
                .astype('float32'))
        self.b_o = theano.shared(
            value=np.zeros((nc,)).astype('float32'))

        self.w_h = theano.shared(name='wh',
            value=he_normal((na*de,nh)).astype('float32'))
        self.b_h = theano.shared(
            value=np.zeros((nh,)).astype('float32'))

        self.attention_w1 = theano.shared(name='att1',
            value=he_normal((de,350)).astype('float32'))

        self.attention_w2 = theano.shared(name='att2',
            value=he_normal((350,na)).astype('float32'))

        self.params = [self.emb, self.w_o, self.b_o, self.attention_w1, self.attention_w2, self.w_h, self.b_h]

        self.eye = theano.shared(name='eye_adv',
            value=np.eye(na).astype('float32'))

        def attention_rec(ht, hat2, mask, eye):
            hat = T.nnet.nnet.softmax(hat2.T)
            p_reg = ((T.dot(hat, hat.T) - self.eye)**2).sum()
            return hat.dot(ht), p_reg


        dropout_switch = T.fscalar('dropout_switch')
        idxs = T.matrix()
        x_word = self.emb[T.cast(idxs, 'int32')]
        mask = T.neq(idxs, 0)*1
        x_word = x_word*mask.dimshuffle(0, 1, 'x')

        Y = T.imatrix('y')

        h_att = T.tanh(T.dot(x_word, self.attention_w1))
        h_att = T.dot(h_att, self.attention_w2)
        mask_att = srng.binomial((x_word.shape[0],na,x_word.shape[1]), p=0.75, dtype='float32')
        [h2, p_reg], updates = theano.scan(attention_rec, sequences=[x_word, h_att, mask_att], non_sequences=[self.eye],
                outputs_info=[None, None])

        p_reg = p_reg.mean()
        h = h2.flatten(2)
        h = rectify(T.dot(h, self.w_h) + self.b_h)
        h = dropout(h, dropout_switch, p_drop)

        pyx = T.nnet.softmax(T.dot(h, self.w_o) + self.b_o)
        pyx = T.clip(pyx, 1e-5, 1-1e-5)

        if train_emb:
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params]) + penalty_p*p_reg
            updates = Adam(L, self.params, lr2=lr, clip=clip)
        else:
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params]) + penalty_p*p_reg
            updates = Adam(L, self.params[1:], lr2=lr, clip=clip)

        self.train_batch = theano.function([idxs, Y, dropout_switch], [L, pyx.argmax(axis=1)], updates=updates, allow_input_downcast=True)
        self.predict_proba = theano.function([idxs, dropout_switch], pyx, allow_input_downcast=True)
        self.predict = theano.function([idxs, dropout_switch], pyx.argmax(axis=1), allow_input_downcast=True)
        self.predict_loss = theano.function([idxs, Y, dropout_switch], [pyx.argmax(axis=1), L], allow_input_downcast=True)

    def __getstate__(self):
        return [x.get_value() for x in self.params]

    def __setstate__(self, weights):
        for x,w in zip(self.params, weights):
            x.set_value(w)
