from tqdm import tqdm
from argparse import ArgumentParser



# read args from the command line
ap = ArgumentParser()
ap.add_argument('train', help='path to the train file')
args = ap.parse_args()


# a function to read the data
def read_conllu(fname):
    data = []
    with open(fname) as f:
        for l in tqdm(f):
            l = l.strip()
            if l == '' or l[0] == '#': # ignore empty and comment lines
                continue

            l = l.split('\t') # split on tabs, some tokens contain spaces
            if '.' in l[0] or '-' in l[0]: # these represent mergers (du = de le) or ellipsis
                continue

            if l[0] == '1': # new sentence starts
                data.append([])

            data[-1].append(l)

    return data


# get the data
train = read_conllu(args.train)
dev = read_conllu(args.train.replace('train', 'dev'))


# statistiques
fcounts = {}
lcounts = {}
ftpcounts = {}
rcounts = {}
for sen in tqdm(train):
    for w in sen:
        form = w[1]
        lem = w[2]
        pos = w[3]

        try:
            fcounts[form] += 1
        except:
            fcounts[form] = 1
        
        try:
            lcounts[lem] += 1
        except:
            lcounts[lem] = 1

        try:
            ftpcounts[form, pos] += 1
        except:
            ftpcounts[form, pos] = 1


        f = '$'+form+'$'
        for i in range(min(3, len(form))):
            r = f[:i+2]
            try:
                rcounts[r, pos] += 1
            except:
                rcounts[r, pos] = 1

            r = f[-i-2:]
            try:
                rcounts[r, pos] += 1
            except:
                rcounts[r, pos] = 1
            
            

forms = sorted(fcounts.items(), key=lambda x: x[1])
lems = sorted(lcounts.items(), key=lambda x: x[1])

# most frequent tokens and lemmas
print(forms[-10:])
print(lems[-10:])


# Now, the actual fun


def eval_pred(pred, gold):
    tp, tn, fp, fn = 0, 0, 0, 0
    for x, y in zip(pred, gold):
        if x == y:
            tp += 1
        elif x == 'UNK':
            fn += 1
        else:
            fp += 1

        # there are no true negative since every word has a POS tag

    print('Precision :', tp/(tp+fp)*100, 'Rappel :', tp/(tp+fn)*100, 'F1 :', 2*tp/(2*tp+fp+fn)*100, '#Prediction :', tp+fp, 'Total :', tp+fp+fn, sep='\t')

gold = [] # get the actual gold truth
for sen in dev:
    for w in sen:
        gold.append(w[3])


# 1. 10 rules on whole forms
# I cheated a bit
#print(sorted(ftpcounts.items(), key=lambda x:x[1])[-10:])
#[(('que', 'SCONJ'), 1814), (('la', 'DET'), 1890), (('li', 'DET'), 1916), (('a', 'ADP'), 2152), (('si', 'ADV'), 2199), (('de', 'ADP'), 3113), (('il', 'PRON'), 3372), (('et', 'CCONJ'), 4105), (('.', 'PUNCT'), 5957), ((',', 'PUNCT'), 11315)] # on fro_profiterole_train   from UD 2.14

ftp = {',':'PUNCT', '.':'PUNCT', 'et':'CCONJ', 'il':'PRON', 'de':'ADP',
       'si':'ADV', 'a':'ADP', 'li':'DET', 'la':'DET', 'que':'SCONJ'}

pred = [] # get the actual gold truth
for sen in dev:
    for w in sen:
        try:
            pred.append(ftp[w[1]])
        except:
            pred.append('UNK')

print(len(pred), len(gold)) # check if length match

eval_pred(pred, gold)


# 2.
#print(sorted(rcounts.items(), key=lambda x:x[1])[-15:]) # some may be redundant
rules = [(',', 'eq', 'PUNCT'), ('.', 'eq', 'PUNCT'), ('et', 'eq', 'CCONJ'),
         ('$l', 'in', 'DET', ),('t$', 'in', 'VERB'), ('s$','in','NOUN'),
         ('$d', 'in', 'ADP'), ('e$', 'in', 'NOUN'), ('l$', 'in', 'PRON'), ('i$', 'in', 'PRON')]
#[(('l$', 'PRON'), 4328), (('i$', 'PRON'), 4468), (('$e', 'CCONJ'), 4751), (('$d', 'ADP'), 4797), (('nt$', 'VERB'), 4798), (('e$', 'PRON'), 5383), (('t$', 'CCONJ'), 5565), (('s$', 'NOUN'), 5683), (('.$', 'PUNCT'), 5957), (('$.', 'PUNCT'), 6027), (('e$', 'NOUN'), 7462), (('$l', 'DET'), 7605), (('$,', 'PUNCT'), 11315), ((',$', 'PUNCT'), 11315), (('t$', 'VERB'), 12080)]

pred2 = []
for sen in dev:
    for w in sen:
        # assume we match the first rule and then break
        pos = 'UNK'
        form = w[1]
        for s, op, p in rules:
            if op == 'eq' and s == form:
                pos = p
                break
            elif op == 'in' and s in '$'+form+'$':
                pos = p
                break

        pred2.append(pos)

print(len(pred2), len(gold)) # check if length match

eval_pred(pred2, gold)
