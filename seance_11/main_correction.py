"""
TAL² 
CPES SDAC
2024-2025
"""
from random import randint, seed
from os import scandir
import torch
from torch import Tensor, nn
from tqdm import tqdm, trange
import numpy as np

from Models import mlp, cnn, rnn

"""
I recommend 
Home Grown Feat. CHEHON - Soul of Moon - DO THE REGGAE!
to do this practical!
"""

K = 500

seed(0)

# reading the data
path = '.'
data = {}
for fl in scandir(path): # put the 
    if fl.is_file() and   '.' not in fl.name and '~' not in fl.name:
        with open(fl) as f:
            text = ''
            for l in f:
                l = l.strip()
                if l == '':
                    continue
                text += l

            data[fl.name] = text

if len(data) == 0:
    print('You need to edit the path variable to match the place where the data are stored.')
    exit()


# get the characters, and the languages
# Truly, there are libraries to do all of this, but learning to do them by hand is good, you understand what you do
chars = set()
for lng, text in data.items():
    #print(x, len(y))
    chars.update(text)

chars = sorted(chars)
ioc = {c:i for i,c in enumerate(chars)} # that's a home made character based tokenizer !
nchar = len(chars)

lngs = sorted(data)
iol = {l:i for i,l in enumerate(lngs)} # turns a language iso into an int
nclass = len(iol)


# train, test, split
train = {x:y[:-500] for x,y in data.items()}
test = {x:y[-500:] for x,y in data.items()}


# let's prepare a standardize train test so that we can compare the results across models
k = 15

XY_train = []
for _ in trange(K):
    for lng, text in train.items():
        # pick a random position is the text
        r = randint(0, len(text) - k)

        XY_train.append((text[r:r+k], lng))


XY_test = []
for _ in trange(K):
    for lng, text in test.items():
        # pick a random position is the text
        r = randint(0, len(text) - k)

        XY_test.append((text[r:r+k], lng))






# things we can do with raw text data:
# predict the language from a sequence of characters


# train an mlp
# you need :
model = mlp(nchar, nclass)      # a model
floss = nn.CrossEntropyLoss()   # a loss function 
trainer = torch.optim.Adam(model.parameters())    # an optimizer

for txt, lng in tqdm(XY_train):
    # prepare the data
    # in practice you'd prepare the data much before you train in a separate loop
    y = iol[lng]
    x = [0] * nchar
    for c in txt:
        x[ioc[c]] += 1

    #print(x, y)
    # empty the gradient
    trainer.zero_grad()
    #compute the score for each langage
    scores = model(Tensor([x]))
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # compute the loss according the true language y
    loss = floss(scores, Tensor([y]).long())
    # compute the gradient
    loss.backward()
    # make a step
    trainer.step()


# test
trainer.zero_grad()
confusion = np.zeros((nclass, nclass))

for text, lng in tqdm(XY_test):
    # prepare the data
    y = iol[lng]
    x = [0] * nchar
    for c in text:
        x[ioc[c]] += 1


    #print(x, y)
    # get the scores
    scores = model(Tensor([x]))
    # take the argmax
    yhat = scores.argmax().item()
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # do they match or not ?
    confusion[y, yhat] += 1

print('MLP')

print('..', '\t'.join(lngs), sep='\t')
for i,lng in enumerate(lngs):
    print(lng, *confusion[i], sum(confusion[i]), sep='\t')

print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')
print('Acc: ', np.trace(confusion) / confusion.sum())

print()

# Analyse the above code and try to understand what is happening

# modify it, increase the size of the windows we want to classify, change the depth/width of the MLP, change the non linearities, train for longer...

# then look at the CNN try to understand it and see if it works better

# eventually, try to implement an RNN∕LSTM∕GRU in the same fashion


# train a CNN
# you need :
model = cnn(nchar, 50, nclass)      # a model
floss = nn.CrossEntropyLoss()   # a loss function 
trainer = torch.optim.Adam(model.parameters())    # an optimizer

for txt, lng in tqdm(XY_train):
    # prepare the data
    y = iol[lng]
    x = [ioc[c] for c in txt] # this is not a vector of count, but a sequence of indices


    #print(x, y)
    # empty the gradient
    trainer.zero_grad()
    #compute the score for each langage
    scores = model(Tensor([x]).long())
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # compute the loss according the true language y
    #print(scores)
    loss = floss(scores, Tensor([y]).long())
    # compute the gradient
    loss.backward()
    # make a step
    trainer.step()


# test
trainer.zero_grad()
confusion = np.zeros((nclass, nclass))

for text, lng in tqdm(XY_test):
    # prepare the data
    y = iol[lng]
    x = [ioc[c] for c in text] # this is not a vector of count, but a sequence of indices

    #print(x, y)
    # get the scores
    scores = model(Tensor([x]).long())
    # take the argmax
    yhat = scores.argmax().item()
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # do they match or not ?
    confusion[y, yhat] += 1

print('CNN')

print('..', '\t'.join(lngs), sep='\t')
for i,lng in enumerate(lngs):
    print(lng, *confusion[i], sum(confusion[i]), sep='\t')

print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')
print('Acc: ', np.trace(confusion) / confusion.sum())

print()




# train a RNN
# you need :
model = rnn(nchar, 50, nclass)      # a model
floss = nn.CrossEntropyLoss()   # a loss function 
trainer = torch.optim.Adam(model.parameters())    # an optimizer

# train the RNN for 5 epochs check what happens

for _ in range(5):
    for txt, lng in tqdm(XY_train):
        # prepare the data
        y = iol[lng]
        x = [ioc[c] for c in txt] # this is not a vector of count, but a sequence of indices
        

        #print(x, y)
        # empty the gradient
        trainer.zero_grad()
        #compute the score for each langage
        scores = model(Tensor([x]).long())
        #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
        # compute the loss according the true language y
        #print(scores)
        loss = floss(scores, Tensor([y]).long())
        # compute the gradient
        loss.backward()
        # make a step
        trainer.step()
        
        
    # test
    trainer.zero_grad()
    confusion = np.zeros((nclass, nclass))
    
    for text, lng in tqdm(XY_test):
        # prepare the data
        y = iol[lng]
        x = [ioc[c] for c in text] # this is not a vector of count, but a sequence of indices
        
        #print(x, y)
        # get the scores
        scores = model(Tensor([x]).long())
        # take the argmax
        yhat = scores.argmax().item()
        #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
        # do they match or not ?
        confusion[y, yhat] += 1

    print('RNN')

    print('..', '\t'.join(lngs), sep='\t')
    for i,lng in enumerate(lngs):
        print(lng, *confusion[i], sum(confusion[i]), sep='\t')
        
    print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')
    print('Acc: ', np.trace(confusion) / confusion.sum())

    print()
