# -*-coding: utf-8-*-
import nltk

# Si nécessaire, télécharger punkt_tab
#nltk.download('punkt_tab')

def charge_corpus(filename):
    ''' retourne une liste de phrases
    qui sont elles-même des listes de tokens'''
    corpus = []
    with open(filename, "r") as f:
        for p in nltk.tokenize.sent_tokenize(f.read()):
            corpus.append(nltk.word_tokenize(p.lower()))
    return corpus

def cumule_contextes(corpus, taille):
    cumul = {}
    for phrase in corpus:
        for i, tk in enumerate(phrase):
            d = max(0, i-taille)
            f = min(i+taille, len(phrase))
            contexte = phrase[d:i] + phrase[i+1:f]
            if tk not in cumul.keys():
                cumul[tk] = contexte
            cumul[tk].extend(contexte)
    return cumul

c = cumule_contextes(charge_corpus("Candide.txt"))



