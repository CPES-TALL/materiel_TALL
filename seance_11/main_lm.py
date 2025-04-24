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

from Models import RNN_LM

"""
I recommend 
Home Grown Feat. CHEHON - Soul of Moon - DO THE REGGAE!
to do this practical!
"""

k = 50

seed(0)

# reading the data
path = '.'
data = {}
for fl in scandir(path):
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
nlang = len(iol)


# train, test, split
train = {x:y[:-500] for x,y in data.items()}
test = {x:y[-500:] for x,y in data.items()}


# things we can do with raw text data:
# predict the next character in a sequence a.k.a language modeling


# train a RNN
# you need :
model = RNN_LM(nchar, 100, nlang, 100, 200)      # a model
floss = nn.CrossEntropyLoss()   # a loss function 
trainer = torch.optim.Adam(model.parameters())    # an optimizer

# train the RNN for 5 epochs check what happens

for _ in range(5):
    model.train()
    for _ in trange(200):
        for lng, txt in train.items():
            #if lng not in ['en', 'es']:
            #    continue
            # prepare the data
            l = iol[lng]
            chs = [ioc[c] for c in txt]

            r = randint(10, len(chs)-k)

            # empty gradient
            trainer.zero_grad()
            # warm the model with 10 characters
            #print(l, chs[r-10], nlang, nchar)
            _, h = model(l, chs[r-10])
            for ch in chs[r-9:r]:
                #print(l, ch)
                _, h = model(l, ch, h)

            loss = 0
            #print()
            for ch, nch in zip(chs[r:r+k-1], chs[r+1:r+k]): # zip the list of char index with itself shifted by one char
                scores, h = model(l, ch, h)
                am = scores[0].argmax()
                #print(am, chars[am.item()], chars[nch], sep='\t')
                loss += floss(scores, Tensor([nch]).long()) # accumulate the score for 99 steps

            #print(loss)
            # compute the gradient
            loss.backward()
            # make a step
            trainer.step()

        
    # test
    trainer.zero_grad()
    model.eval()

    print('RNN')
    for lng, text in train.items():
        tot = 0
        good = 0
        # prepare the data
        l = iol[lng]
        chs = [ioc[c] for c in text]

        # warming
        scores, h = model(l, chs[0])
        for ch in chs[:10]:
            #print(l, ch)
            _, h = model(l, ch, h)

        for ch, nch in zip(chs[10:], chs[11:]): # zip the list of char index with itself shifted by one char
            # get the scores
            scores, h = model(l, ch, h)
            # take the argmax
            yhat = scores.argmax().item()

            #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
            # do they match or not ?
            tot += 1
            if nch == yhat:
                good += 1

        """
        print('..', '\t'.join(lngs), sep='\t')
        for i,lng in enumerate(lngs):
        print(lng, *confusion[i], sum(confusion[i]), sep='\t')
        
        print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')
        print('Acc: ', np.trace(confusion) / confusion.sum())
        """
        print(lng, 'Acc: ', good/tot)


    print('test')
    for lng, text in test.items():
        tot = 0
        good = 0
        # prepare the data
        l = iol[lng]
        chs = [ioc[c] for c in text]

        # warming
        _, h = model(l, chs[0])
        for ch in chs[:10]:
            #print(l, ch)
            _, h = model(l, ch, h)

        for ch, nch in zip(chs[10:], chs[11:]): # zip the list of char index with itself shifted by one char
            # get the scores
            scores, h = model(l, ch, h)
            # take the argmax
            yhat = scores.argmax().item()

            #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
            # do they match or not ?
            tot += 1
            if nch == yhat:
                good += 1
                
            
        """
        print('..', '\t'.join(lngs), sep='\t')
        for i,lng in enumerate(lngs):
        print(lng, *confusion[i], sum(confusion[i]), sep='\t')
        
        print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')
        print('Acc: ', np.trace(confusion) / confusion.sum())
        """
        print(lng, 'Acc: ', good/tot)
        
    print()
