from tqdm import tqdm
from argparse import ArgumentParser
from random import choice, randint, shuffle



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


def eval_pred(pred, gold, verbose=False):
    tp, tn, fp, fn = 0, 0, 0, 0
    for x, y in zip(pred, gold):
        if x == y:
            tp += 1
        elif x == 'UNK':
            fn += 1
        else:
            fp += 1

        # there are no true negative since every word has a POS tag

    if verbose:
        #print('Precision :', tp/(tp+fp+1)*100, 'Rappel :', tp/(tp+fn+1)*100, 'F1 :', 2*tp/(2*tp+fp+fn+1)*100, '#Prediction :', tp+fp, 'Total :', tp+fp+fn, sep='\t')
        tqdm.write('\t'.join(str(x) for x in ('Precision :', tp/(tp+fp+1)*100, 'Rappel :', tp/(tp+fn+1)*100, 'F1 :', 2*tp/(2*tp+fp+fn+1)*100, '#Prediction :', tp+fp, 'Total :', tp+fp+fn)))
    return 2*tp/(2*tp+fp+fn)*100
    

tgold = [] # get the actual gold truth
for sen in train:
    for w in sen:
        tgold.append(w[3])


gold = [] # get the actual gold truth
for sen in dev:
    for w in sen:
        gold.append(w[3])



# Now, the actual fun

def sample_rule(data):
    sen = choice(data) # pick a sentence at random

    scheme = choice(['*f*', '*f*', '*f*', '*f*', # i repeat some to bias the probability of picking them
                     'pf*', 'ff*', 'pf*', 'ff*',
                     '*fp', '*ff', '*fp', '*ff',
                     'f*f', 'p*p', 'f*p', 'p*f',
                     'fff', 'pff', 'ffp', 'pfp']) # f is form, p is pos, * is no rule we assign a pos to the middle one

    wid = randint(0, len(sen)-1) 
    
    rule = []
    for i, x in enumerate(scheme): # turn the scheme into an actual rule
        if x == '*':
            rule.append(x)
        elif wid + i - 1 < 0 or wid + i - 1 >= len(sen): # outside of sentence
            rule.append('$$$')
        elif x == 'p':
            rule.append(sen[wid+i-1][3])
        elif x == 'f':
            form = sen[wid+i-1][1]
            l = randint(1, min(3, len(form)))
            if choice('lr') == 'l':
                rule.append('$'+form[:l])
            else:
                rule.append(form[-l:]+'$')

    return scheme, tuple(rule), sen[wid][3] # return the rule
    


# genetic shenanigans


def apply_tagger(tagger, data):
    pred = []
    for sen in dev: # for each sentence
        postags = ['UNK' for _ in sen] # initialize POS

        for scheme, (x, y, z), pos in tagger: # for each rule in turn
            for i, w in enumerate(sen):
                if postags[i] != 'UNK': # we have already assigned a pos, but we could use the last rule, or make the rules vote
                    # fun stuff, if we use the last rule, we could have intermediate values that are not POS tags, and we could in theory make very complicated systems
                    continue

                match = [False, False, False]
                if y == '*' or y in '$'+w[1]+'$': # we're matching the taget word
                    match[1] = True

                if x == '*': # match the left word
                    match[0] = True
                elif i == 0:
                    if x == '$$$':
                        match[0] = True
                elif scheme[0] == 'p' and postags[i-1] == x:
                    match[0] = True
                elif scheme[0] == 'f' and x in '$'+sen[i-1][1]+'$':
                    match[0] = True

                if z == '*': # match the right word
                    match[2] = True
                elif i == len(sen)-1:
                    if z == '$$$':
                        match[2] = True
                elif scheme[2] == 'p' and postags[i+1] == z:
                    match[2] = True
                elif scheme[2] == 'f' and z in '$'+sen[i+1][1]+'$':
                    match[2] = True

                if match == [True, True, True]:
                    postags[i] = pos

        pred += postags

    return pred


# now the real real fun

pop = []
for _ in range(20): # initialise the algo with 20 taggers
    tagger = [sample_rule(train) for _ in range(15)] # a tagger is 15 random rules
    tagger.sort(key=lambda x:'p' in x[0])

    f1 = eval_pred(apply_tagger(tagger, train), tgold, True) # test each on the train set
    pop.append((f1, tagger))

    
for _ in range(50): # for 50 iterations
    pop.sort() # sort the taggers according to their f1 score

    for _, tagger in tqdm(pop[-10:], leave=False, position=0): # for the best 10 parsers
        new_tagger = [t for t in tagger] # create a new one, by copy and shuffling the rules
        shuffle(new_tagger)
        
        new_tagger.append(sample_rule(train)) # add a new rule
        new_tagger.sort(key=lambda x:'p' in x[0]) # sort the rules so that those using predicted pos arrive at the end

        f1 = eval_pred(apply_tagger(new_tagger, train), tgold, True) # test
        pop.append((f1, new_tagger))
    
        # rinse and repeat
        # we could also remove rules from time to time
        # at random, or by scoring individual rules and picking the least useful ones

        # lotsa fun to be have 
