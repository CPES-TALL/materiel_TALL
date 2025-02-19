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


