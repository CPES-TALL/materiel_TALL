from os import scandir
from tqdm import tqdm
from numpy import zeros
from numpy.linalg import norm
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from math import log


# load the data
data = {}
fls = [x.name for x in scandir('pages') if '~' not in x.name] # the list of pages
fls.sort()

for fname in fls: # for each page
    data[fname] = []
    fl = open('pages/'+fname) # open it # in windows/max you may need to set encoding='utf-8'

    for l in fl: # read line by line
        l = l.strip()
        if l == '':
            continue

        data[fname].append(l.split())
    #print(fname, len(data[fname]))

# get the vocabulary
voc = {}
for fname, content in data.items():
    for line in content:
        for w in line:
            try:
                voc[w] += 1
            except:
                voc[w] = 1

print('Size voc before threshold :', len(voc))
voc = sorted([k for k, n in voc.items() if n > 25])
voc = sorted(voc)
print('After :', len(voc))

# create a pair of dictionaries : int of file and int of word, that are used as reference indices
iof = {fname:i for i, fname in enumerate(sorted(fls))}
iow = {w:i for i,w in enumerate(voc)}


# now the matrix
vecs = zeros((len(data), len(voc)))

for fname, content in data.items():
    for line in content:
        for w in line:
            # for each word of each line of each file, if it has an index, count it
            try:
                vecs[iof[fname], iow[w]] += 1
            except:
                ()

# now reduce dimension
pca = PCA(2)
X = pca.fit_transform(vecs)

#print('\t'.join(voc))

# most relevant feature for each component
for i, comp in enumerate(pca.components_):
    comps = sorted([(c, j) for j, c in enumerate(comp)])
    for c, j in comps[:10] + comps[-10:]:
        print(i, voc[j], c, j, sep='\t')

    print()


#for i, x in enumerate(X):
#    print(fls[i], ' '.join(str(y) for y in x))

fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1])
for i, fn in enumerate(fls):
    ax.annotate(fn, X[i])

plt.savefig('PCA_raw')


# computing cosine for fun
"""
cos = zeros((len(fls), len(fls)))
for i in range(len(data)):
    for j in range(i+1, len(data)):
        cos[i,j] = X[i].dot(X[j]) / norm(X[i]) / norm(X[j])

for i, c in enumerate(cos):
    print(fls[i], '\t'.join(str(x) for x in c))
"""


# tsne
tsne = TSNE(metric="cosine")
X = tsne.fit_transform(vecs)
points = X

fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1])
for i, fn in enumerate(fls):
    ax.annotate(fn, points[i])

plt.savefig('tsne-cosine_raw')



# now normalise
# TF IDF

for i, v in enumerate(vecs):
    t = sum(v)
    for j, k in enumerate(v):
        if k == 0:
            continue
        vecs[i,j] = k / t * log(len(data) / len([x for x in vecs[:,j] if x != 0]))




pca = PCA(2)
X = pca.fit_transform(vecs)

print('\t'.join(voc))

for i, comp in enumerate(pca.components_):
    comps = sorted([(c, j) for j, c in enumerate(comp)])
    for c, j in comps[:10] + comps[-10:]:
        print(i, voc[j], c, j, sep='\t')

    print()

    
for i, x in enumerate(X):
    print(fls[i], ' '.join(str(y) for y in x))

points = X

fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1])
for i, fn in enumerate(fls):
    ax.annotate(fn, points[i])

plt.savefig('PCA_tfidf')


# computing cosine for fun

cos = zeros((len(fls), len(fls)))
for i in range(len(data)):
    for j in range(i+1, len(data)):
        cos[i,j] = vecs[i].dot(vecs[j]) / norm(vecs[i]) / norm(vecs[j])
        cos[j,i] = cos[i,j]

print()
# note : nearest neighbour is not symetrical  : finlande is closer to suede but suede to stockholm and helsinki to finlande
for i, c in enumerate(cos):
    #print(fls[i], '\t'.join(str(x)[:5] for x in c))
    print(fls[i], fls[c.argmax()], sep='\t')



tsne = TSNE(metric="cosine", perplexity=10)
X = tsne.fit_transform(vecs)
points = X

fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1])
for i, fn in enumerate(fls):
    ax.annotate(fn, points[i])

plt.savefig('tsne-cosine_tfidf')
