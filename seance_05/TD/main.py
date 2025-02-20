# -*-coding: utf-8-*-
import nltk
# Si nécessaire, télécharger punkt_tab
#nltk.download('punkt_tab')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def matrice_terme_terme(cumul): 
    contextes = [" ".join(contexte) for contexte in cumul.values()]
    comptages = vectorizer.fit_transform(contextes)
    matrice= pd.DataFrame(comptages.toarray(),
                          index = cumul.keys(),
                          columns = vectorizer.get_feature_names_out())
    return matrice


vectorizer = CountVectorizer()
c = cumule_contextes(charge_corpus("Candide.txt"),2)
mtt = matrice_terme_terme(c)
cosine_sim_matrix = cosine_similarity(mtt)
