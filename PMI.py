import nltk
import math
import random


class PMI:
    def __init__(self, corpus, num_ngrams, window_size):
        """
        Creates a PMI object for a certain corpus. The corpus is
        processed in multiple ways: for every n-gram size, this function
        creates a set of all the n-grams in the corpus, and then finds the
        number of appearences of each element in the set.

        Args:
            corpus(str): The corpus for the PMI
            num_ngrams(int): The range of n-gram sizes (1-num_ngrams)
            window_size(int): Window size for co-occurrence searching
        """

        self.num_ngrams = num_ngrams
        self.window_size = window_size
        self.text = corpus
        self.corpus_ngrams = self.get_corpus_ngrams(corpus)
        self.corpus_ngrams_set = self.make_set()
        self.corpus_ngrams_frequency = self.get_corpus_frequency()
        self.corpus_len = len(self.corpus_ngrams)

    def make_set(self):
        """
        Creates a dictionary that contains a set of evey n-gram in the corpus
        for ever size n-gram

        Returns:
            dict: a dictionary of sets.
        """
        ngrams_set = dict()
        for i, ngrams in self.corpus_ngrams.items():
            ngrams_set[i] = list(set(ngrams))
        return ngrams_set

    def get_corpus_frequency(self):
        """
        Builds a dictionary of n-gram frequencies for all the n-grams in the
        corpus

        Returns:
            dict: dictionary where each size n-gram has a dictionary that maps
                every n-gram of that size to its frequency in the corpus
        """
        freqencies = dict()
        for i in range(1, self.num_ngrams + 1):
            freqencies[i] = dict()
            for ngram in self.corpus_ngrams_set[i]:
                freqencies[i][ngram] = self.get_ngram_frequency(ngram, i)
        return freqencies

    def get_ngram_frequency(self, ngram, len):
        """
        Counts the number of times a n-gram appears in the corpus

        Args:
            ngram(str): a ngram to get the frequency of
            len(int): the size of the n-gram
        Returns:
            int: the number of occurrences in the corpus
        """
        num_occurrences = 0
        for corpus_ngram in self.corpus_ngrams[len]:
            if corpus_ngram == ngram:
                num_occurrences += 1
        return num_occurrences

    def get_corpus_ngrams(self, corpus):
        """
        Creates a list of n-grams in the corpus for every size n-gram
        
        Args:
            corpus(str): The corpus

        Returns:
            dict: for every size n-gram, a list of n-grams in the order of
            the context
        """
        ngrams = dict()
        tokens = nltk.word_tokenize(corpus)
        for i in range(1, self.num_ngrams + 1):
            ngrams[i] = list(nltk.ngrams(tokens, i))
        return ngrams

    def get_qa_ngrams(self, text):
        """
            Takes either the question and answer, and creates a dictionary where
            there is a mapping of every n-gram to its frequency for every size
            n-gram
        Args:
            text(str): Either the question or the answer

        Returns:
            dict: for every size n-gram a mapping of each n-gram to its frequency
        """
        ngrams = dict()
        tokens = nltk.word_tokenize(text)
        for i in range(1, self.num_ngrams+1):
            ngrams[i] = dict()
            for ngram in list(nltk.ngrams(tokens, i)):
                ngrams[i][ngram] = self.get_ngram_frequency(ngram, i)
        return ngrams

    def get_co_occurrence(self, ngram_1, ngram_2, i):
        """
        finds the frequency that 2 n-grams occur together in the corpus
        within the window

        Args:
            ngram_1(str): first n-gram
            ngram_2(str): second n-gram
            i(int): the size of the n-gram

        Returns:
            int: the number of occurrences
        """
        num_matches = 0;
        for j, word in enumerate(self.corpus_ngrams[i]):
            if word == ngram_1[0]:
                for k in range(-self.window_size, self.window_size):
                    if j+k < 0 or j+k >= len(self.corpus_ngrams[i]):
                        continue
                    if ngram_2[0] == self.corpus_ngrams[i][j+k]:
                        num_matches+=1
        return num_matches

    def get_pmi(self, ngram_1, ngram_2, i):
        """
        calculates the pmi score for 2 n-grams. both n-grams must occur in the
        corpus at least once

        Args:
            ngram_1(tuple): a tuple the n-gram, and its frequency
            ngram_2(tuple): a tuple of n-grams, and its frequency
            i(int): the size of the n-gram

        Returns:
            int: pmi score
        """

        bigram_frequency = self.get_co_occurrence(ngram_1, ngram_2, i)

        pmi = (bigram_frequency) / (ngram_1[1] * ngram_2[1])
        return pmi

    #gets the pmi for all the n-grams in the question and answer, and returns the average
    def sentence_pmi_score(self, question_ngrams, answer_ngrams):
        """
        gets the total pmi score by averaging the PMI of each pair of n-grams
        in the question and answer
        Args:
            question_ngrams(dict): dict with all the n-grams with their frequency
            answer_ngrams(dict):  dict with all the n-grams with their frequency

        Returns:
            int: the average PMI score of all the n-grams in the question and
                answer
        """
        pmi = list()
        for i in range(1, self.num_ngrams + 1):
            for ngram_1 in question_ngrams[i].items():
                if ngram_1[1] == 0:
                    pmi.append(0)
                    continue
                for ngram_2 in answer_ngrams[i].items():
                    if ngram_2[1] == 0:
                        pmi.append(0)
                        continue
                    pmi.append(self.get_pmi(ngram_1, ngram_2, i))
        return sum(pmi) / len(pmi)


    def get_result(self, question, answers):
        """
        Finds the average PMI score between the question and each of the answers,
        and returns the result

        Args:
            question(dict): the question
            answers(list): a list of the answers(str)

        Returns:
            dict: the results

        """
        question_text = question["question"]
        results = {"question": question_text}
        results["answer"] = 0
        highest_pmi = 0
        best_answers = []
        question_ngrams = self.get_qa_ngrams(question_text)
        for i, answer in enumerate(answers):
            answer_info = {"id":answer["id"], "text":answer['text']}
            answer_ngrams = self.get_qa_ngrams(answer['text'])
            pmi = self.sentence_pmi_score(question_ngrams, answer_ngrams)
            answer_info["PMI"] = pmi
            if pmi > highest_pmi:
                highest_pmi = pmi
                best_answers = [i]
            elif pmi == highest_pmi:
                best_answers.append(i)
            results["answer_{}".format(i)] = answer_info

        random.shuffle(best_answers)
        results["answer"] = best_answers[0]
        if highest_pmi == 0:
            results["answer"] = -1
        return results
