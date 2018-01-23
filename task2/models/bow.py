import numpy as np
from theano import tensor as T
import theano

from utils import *

class BoW(object):
    ''' Experimental BoW model.
    '''
    def __init__(self, emb, nc=2, de=300, p_drop=0.5, nh=2048,
            penalty=0., train_emb=True, decay=0., clip=None, lr=0.001):
        ''' Init Experimental BoW model.

            Args:
            emb: Word embeddings matrix (num_words x word_dimension)
            nc: Number of classes
            de: Dimensionality of word embeddings
            p_drop: Dropout probability
            nh: Hidden layer dimensions
            penalty: l2 regularization param
            train_emb: Boolean if the embeddings should be trained
            decay: Learning rate decay parameter
            clip: Gradient Clipping parameter (None == don't clip)
            lr: Initial learning rate
        '''

        self.emb = theano.shared(name='Words',
            value=as_floatX(emb))
        self.w_o = theano.shared(
                value=he_normal((nh, nc)))
        self.b_o = theano.shared(
            value=as_floatX(np.zeros((nc,))))

        self.w_h = theano.shared(name='wh',
            value=he_normal((de,nh)))
        self.b_h = theano.shared(
            value=as_floatX(np.zeros((nh,))))

        self.params = [self.emb, self.w_o, self.b_o, self.w_h, self.b_h]

        dropout_switch = T.fscalar('dropout_switch')
        idxs = T.matrix()
        mask = T.neq(idxs, 0)*as_floatX(1.)
        norm = mask.sum(axis=1)

        x_word = self.emb[T.cast(idxs, 'int32')]
        x_word = x_word*mask.dimshuffle(0, 1, 'x')
        x_word = x_word.sum(axis=1)
        x_word = x_word * (as_floatX(1.)/norm.dimshuffle(0, 'x'))
        Y = T.imatrix('y')

        h = rectify(T.dot(x_word, self.w_h) + self.b_h)
        h = dropout(h, dropout_switch, p_drop)

        pyx = T.nnet.softmax(T.dot(h, self.w_o) + self.b_o)
        pyx = T.clip(pyx, as_floatX(1e-5), as_floatX(1-1e-5))

        if train_emb:
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params])
            updates = Adam(L, self.params, lr2=lr, clip=clip, decay=decay)
        else:
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params])
            updates = Adam(L, self.params[1:], lr2=lr, clip=clip, decay=decay)

        self.train_batch = theano.function([idxs, Y, dropout_switch], [L, pyx.argmax(axis=1)], updates=updates, allow_input_downcast=True)
        self.predict_proba = theano.function([idxs, dropout_switch], pyx, allow_input_downcast=True)
        self.predict = theano.function([idxs, dropout_switch], pyx.argmax(axis=1), allow_input_downcast=True)
        self.predict_loss = theano.function([idxs, Y, dropout_switch], [pyx.argmax(axis=1), L], allow_input_downcast=True)

    def __getstate__(self):
        return [x.get_value() for x in self.params]

    def __setstate__(self, weights):
        for x,w in zip(self.params, weights):
            x.set_value(w)
