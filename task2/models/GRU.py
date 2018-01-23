import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

ITERS = 5000
max_features = 20000
batch_size = 40

print 'Loading data ...'

# training data are index for words for the embedding layer,
x_train = np.load('data/X_train_imdb_idx.npy')
y_train = np.load('data/y_train_imdb_idx.npy')

x_test = np.load('data/X_test_imdb_idx.npy')
y_test = np.load('data/y_test_imdb_idx.npy')

class BIGRU(nn.Module):
    def __init__(self):
        super(BIGRU, self).__init__()

        self.embedding = nn.Embedding(max_features, 128)
        self.gru = nn.GRU(128, 64, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(128, 1) # input dim is 64*2 because its bidirectional

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x, h)
        x = self.dropout(x[:,-1,:].squeeze()) # just get the last hidden state
        x = F.sigmoid(self.linear(x)) # sigmoid output for binary classification
        return x, h

    def init_hidden(self):
        return autograd.Variable(torch.randn(2, batch_size, 64)).cuda()


model = BIGRU()

print 'Putting model on GPU ... '
model.cuda()

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_batch_epoch = len(x_train) // batch_size

h = model.init_hidden()

print 'Training ...'
model.train()
for e in range(ITERS):
    print '\n' + 'Epoch {}/{}'.format(e, ITERS)
    print '-' * 10
    start = time.time()

    idx = np.random.permutation(len(x_train))
    iter_loss = 0.
    iter_correct = 0.
    for b in range(len(x_train) // batch_size):
        x_b = torch.from_numpy(x_train[idx[b*batch_size:(b+1)*batch_size]]).long()
        y_b = torch.from_numpy(y_train[idx[b*batch_size:(b+1)*batch_size]]).float()

        x_batch = autograd.Variable(x_b.cuda())#
        y_batch = autograd.Variable(y_b.cuda())#

        h.detach_()
        y_pred, h = model(x_batch, h)

        optimizer.zero_grad()

        loss = loss_fn(y_pred, y_batch)
        loss.backward()

        #clipped_lr = 0.001 * nn.utils.clip_grad_norm(model.parameters(), 0.25)
        # clip gradients because RNN
        #for p in model.parameters():
        #    p.data.add_(-clipped_lr, p.grad.data)

        optimizer.step()

        trn_preds = torch.round(y_pred.data)
        iter_correct += torch.sum(trn_preds == y_batch.data)
        iter_loss += loss.data[0]

    print 'Training Loss: {:.3} | Training Acc: {:.3}'.format(iter_loss / num_batch_epoch, float(iter_correct) / num_batch_epoch)
    print 'Time: {}'.format(time.time()-start)

print '\n' + 'Evaluating network ...'
model.eval() # changes the behavior of the dropout for making predictions
# evaluate the accuracy
running_corrects = 0
for vb in range(len(x_test) // batch_size):
    x_v_b = torch.from_numpy(x_test[vb*batch_size:(vb+1)*batch_size]).long()
    y_v_b = torch.from_numpy(y_test[vb*batch_size:(vb+1)*batch_size]).float()

    x_v_batch = autograd.Variable(x_v_b.cuda(), volatile=True)
    y_v_batch = autograd.Variable(y_v_b.cuda())
    outputs, h = model(x_v_batch, h)
    preds = torch.round(outputs.data)
    running_corrects += torch.sum(preds == y_v_batch.data)

    if vb % 100 == 0:
        print 'Valid batch:', vb, 'done!'

print 'Percent correct:', float(running_corrects) / len(x_test)