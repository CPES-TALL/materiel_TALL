import nltk

# Si nécessaire, télécharger punkt_tab
nltk.download('punkt_tab')


with open("Candide.txt", "r") as f:
    raw = f.read()

phrases = nltk.tokenize.sent_tokenize(raw)

corpus = []
for p in phrases:
    tkl = nltk.word_tokenize(p)
    corpus.append(tkl)

# Corpus : liste de phrases
# Phrase(s) : liste de tokens
# Token(s): string

from collections import Counter, defaultdict

def construit_modele(corpus, n=2):
    modele = defaultdict(Counter)
    for phrase in corpus:
        phrase = ['<s>'] + phrase + ['</s>']
        for i in range(len(phrase) - n + 1):
            contexte, mot = tuple(phrase[i:i+n-1]), phrase[i+n-1]
            modele[contexte][mot] += 1
    return modele

modele = construit_modele(corpus)

# Exploitation du modèle
# On n'est pas obligés de convertir en probas en fait: on prend le max.

def prediction(m, contexte, n=2):
    '''prédit le mot qui doit suivre en prenant
    les derniers n-1 mots du contexte (si possible)'''
    # à écrire
