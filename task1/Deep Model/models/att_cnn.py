from theano import tensor as T
import theano
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d as max_pool_2d
from utils import *
class CNN(object):
    ''' CNN Model (http://www.aclweb.org/anthology/D14-1181)
    '''
    def __init__(self, emb, nf=400, nc=2, de=400, p_drop=0.5, fs=[3,4,5], penalty=0,
            lr=0.001, decay=0., clip=None, train_emb=True, att=8):
        ''' Init Experimental CNN model.
            Args:
            emb: Word embeddings matrix (num_words x word_dimension)
            nc: Number of classes
            de: Dimensionality of word embeddings
            p_drop: Dropout probability
            fs: Convolution filter width sizes
            penalty: l2 regularization param
            lr: Initial learning rate
            decay: Learning rate decay parameter
            clip: Gradient Clipping parameter (None == don't clip)
            train_emb: Boolean if the embeddings should be trained
        '''
        self.emb = theano.shared(name='Words',
            value=as_floatX(emb))
        self.filter_w = []
        self.filter_b = []
        for filter_size in fs:
            self.filter_w.append(theano.shared(
                value=he_normal((nf, 1, filter_size, de))
                .astype('float32')))
            self.filter_b.append(theano.shared(
                value=np.zeros((nf,)).astype('float32')))
        #value=he_normal((nf*len(fs)+de+8*de, nc))
        #value=he_normal((8*de, nc))
        self.w_o = theano.shared(
            value=he_normal((att*400, nc))
                .astype('float32'))
        self.b_o = theano.shared(
            value=as_floatX(np.zeros((nc,))))
        self.attention_w1 = theano.shared(name='att1',
            value=he_normal((de,128)).astype('float32'))
        self.attention_w2 = theano.shared(name='att2',
            value=he_normal((128,att)).astype('float32'))
        self.params = [self.emb, self.w_o, self.b_o, self.attention_w1, self.attention_w2]
        for w, b in zip(self.filter_w, self.filter_b):
            self.params.append(w)
            self.params.append(b)
        dropout_switch = T.fscalar('dropout_switch')
        idxs = T.matrix()
        x_word = self.emb[T.cast(idxs.flatten(), 'int32')].reshape((idxs.shape[0], 1, idxs.shape[1], de))
        x_word2 = self.emb[T.cast(idxs, 'int32')]
        mask = T.neq(idxs, 0)*as_floatX(1.)
        x_word = x_word*mask.dimshuffle(0, 'x', 1, 'x')
        x_word2 = x_word2*mask.dimshuffle(0, 1, 'x')
        Y = T.imatrix('y')
        l1_w_all = []
        for w, b, width in zip(self.filter_w, self.filter_b, fs):
            l1_w = conv2d(x_word, w, image_shape=(None,1,None,de), filter_shape=(nf, 1, width, de))
            l1_w = rectify(l1_w+ b.dimshuffle('x', 0, 'x', 'x'))
            #l1_w = T.max(l1_w, axis=2).flatten(2)
            l1_w = T.max(l1_w, axis=2)
            l1_w = l1_w.reshape((l1_w.shape[0], 1, nf))
            l1_w_all.append(l1_w)
        norm = mask.sum(axis=1)
        word_avg = x_word2.sum(axis=1)
        word_avg = word_avg * (as_floatX(1.)/norm.dimshuffle(0, 'x'))
        l1_w_all.append(word_avg.reshape((idxs.shape[0],1,de)))
        new_x_words = x_word2
        l1_w_all.append(new_x_words)
        l1 = T.concatenate(l1_w_all, axis=1)
        def attention_rec(ht, hat2):
            hat = T.nnet.nnet.softmax(hat2.T)
            return hat.dot(ht)
        h_att = T.tanh(T.dot(l1, self.attention_w1))
        h_att = T.dot(h_att, self.attention_w2)
        h2, updates = theano.scan(attention_rec, sequences=[l1, h_att],
                outputs_info=[None])
        #h2 = h2.flatten(2)
        final_act = []
        cnter = 0
        h = h2.flatten(2)
        h = dropout(h, dropout_switch, p_drop)
        pyx = T.nnet.softmax(T.dot(h, self.w_o) + self.b_o)
        if train_emb:
            #L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params])
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([abs(p).sum() for p in self.params])
            updates = Adam(L, self.params, lr2=lr, clip=clip)
        else:
            L = T.nnet.nnet.categorical_crossentropy(pyx, Y).mean() + penalty*sum([(p**2).sum() for p in self.params])
            updates = Adam(L, self.params[1:], lr2=lr, clip=clip)
        self.train_batch = theano.function([idxs, Y, dropout_switch], [L, pyx.argmax(axis=1)], updates=updates, allow_input_downcast=True)
        self.predict = theano.function([idxs, dropout_switch],
                outputs=pyx.argmax(axis=1), allow_input_downcast=True)
        self.predict_proba = theano.function([idxs, dropout_switch], outputs=pyx, allow_input_downcast=True)
        self.predict_loss = theano.function([idxs, Y, dropout_switch], [pyx.argmax(axis=1), L], allow_input_downcast=True)
    def __getstate__(self):
        data = [x.get_value() for x in self.params]
        return data
    def __setstate__(self, data):
        for w1, w2 in zip(self.params, data):
            w1.set_value(w2)
