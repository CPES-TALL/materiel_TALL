from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

with open("Candide.txt", "r") as f:
    raw = f.read()

phrases = sent_tokenize(raw)

corpus = []
for p in phrases:
    tkl = word_tokenize(p)
    corpus.append(tkl)

# Corpus : liste de phrases
# Phrase(s) : liste de tokens
# Token(s): string
