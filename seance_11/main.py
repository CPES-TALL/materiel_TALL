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

from Models import mlp

seed(0)

# reading the data
data = {}
for fl in scandir('.'):
    if fl.is_file() and   '.' not in fl.name and '~' not in fl.name:
        with open(fl) as f:
            text = ''
            for l in f:
                l = l.strip()
                if l == '':
                    continue
                text += l

            data[fl.name] = text


# get the characters
# Truly, there are libraries to do all of this, but learning to do them by hand is good, you understand what you do
chars = set()
for x, y in data.items():
    #print(x, len(y))
    chars.update(y)

chars = sorted(chars)
ioc = {c:i for i,c in enumerate(chars)} # that's a home made character based tokenizer !
nchar = len(chars)

lngs = sorted(data)
iol = {l:i for i,l in enumerate(lngs)}
nclass = len(iol)


# train, test, split
train = {x:y[:-500] for x,y in data.items()}
test = {x:y[-500:] for x,y in data.items()}
    

# things we can do with raw text data:
# predict the language from a sequence of characters


# train an mlp
model = mlp(nchar, nclass)
floss = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(model.parameters())

k = 10
for _ in trange(100):
    for lng, text in train.items():
        # pick a random position is the text
        r = randint(0, len(text) - k)

        # prepare the data
        # in practice you'd prepare the data much before you train in a separate loop
        y = iol[lng]
        x = [0] * nchar
        for i in range(k):
            x[ioc[text[i+r]]] += 1


        #print(x, y)
        # empty the gradient
        trainer.zero_grad()
        scores = model(Tensor([x]))
        print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
        loss = floss(scores, Tensor([y]).long())
        loss.backward()
        trainer.step()


# test
trainer.zero_grad()
confusion = np.zeros((nclass, nclass))

for _ in trange(100):
    for lng, text in train.items():
        # pick a random position is the text
        r = randint(0, len(text) - k)

        # prepare the data
        y = iol[lng]
        x = [0] * nchar
        for i in range(k):
            x[ioc[text[i+r]]] += 1


        #print(x, y)
        # empty the gradient
        scores = model(Tensor([x]))
        yhat = scores.argmax().item()
        print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
        confusion[y, yhat] += 1


print('..', '\t'.join(lngs), sep='\t')
for i,lng in enumerate(lngs):
    print(lng, *confusion[i], sum(confusion[i]), sep='\t')

print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')


# Analyse the above code and try to understand what is happening

# modify it, increase the size of the windows we want to classify, change the depth/width of the MLP, change the non linearities, train for longer...

# then look at the CNN try to understand it and see if it works better

# eventually, try to implement an RNN∕LSTM∕GRU in the same fashion


