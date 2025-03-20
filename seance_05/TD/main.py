# -*-coding: utf-8-*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# fonction: charge_corpus()
# in : nom de fichier (string)
# out: liste de phrases segmentées et tokenisées par nltk
#      phrase = liste de tokens (string)
# import nltk
# potentiellement installation de punkt_tab
# ------------------------------------------------------------
import nltk
# Si nécessaire, télécharger punkt_tab
#nltk.download('punkt_tab')

def charge_corpus(filename):
    corpus = []
    with open(filename, "r") as f:
        for p in nltk.tokenize.sent_tokenize(f.read()):
            corpus.append(nltk.word_tokenize(p.lower()))
    return corpus

# ------------------------------------------------------------
# fonction: cumule_voisins()
# in : corpus (liste de listes de tokens)
#      demi-taille de la fenêtre (k) (int)
# out: dictionnaire {
#         clés = tokens
#         valeurs = listes de tous les tokens du voisinage
#      }
# ------------------------------------------------------------
def cumule_voisins(corpus, k):
    cumul = {}
    for phrase in corpus:
        for i, tok in enumerate(phrase):
            d = max(0, i-k)
            f = min(i+k, len(phrase))
            voisins = phrase[d:i] + phrase[i+1:f]
            if tok not in cumul.keys():
                cumul[tok] = voisins
            cumul[tok].extend(voisins)
    return cumul

def stats_cumul(c):
    print("Vocabulaire : %d types" % len(c))
    

c = cumule_voisins(charge_corpus("Candide.txt"),2)

stats_cumul(c)



'''









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

mot = 'monsieur'

indice = vectorizer.vocabulary_.get(mot)

cible_similarite = cosine_sim_matrix[indice]

dix_voisins_indices = np.argsort(cible_similarite)[::-1][1:11]

# Get the words corresponding to the closest neighbors
closest_neighbors = [list(vectorizer.vocabulary_.keys())[i] for i in dix_voisins_indices]

# Display the 10 closest neighbors
print("Les  10 voisins les plus proches de '%s' sont %s"  % (mot, closest_neighbors))

'''
