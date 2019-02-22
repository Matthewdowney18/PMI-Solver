import nltk

class PMI:
    def __init__(self, corpus):

        self.corpus = nltk.tokenize.word_tokenize(corpus)
        self.corpus_words = list(set(self.corpus))

