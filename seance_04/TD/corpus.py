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
