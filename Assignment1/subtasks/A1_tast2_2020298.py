import re
from pathlib import Path
import string
from functools import reduce
from math import log
import itertools
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from transformers import pipeline

classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True,)

def emotion_scores(sample):
    emotion=classifier(sample)
    return emotion[0]

filename = "corpus.txt"

def load_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    print("No of sentences in Corpus: "+str(len(lines)))
    return lines

class BigramLM:
    def __init__(self):
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.vocab = set()
        self.bigrams = set()

    def learn_model(self, dataset):
      for sentence in dataset:
          tokens = sentence.split()
          for i in range(1, len(tokens)):
              bigram = (tokens[i-1], tokens[i])
              self.bigram_counts[bigram] += 1
              self.unigram_counts[tokens[i-1]] += 1
              self.vocab.add(tokens[i-1])
              self.vocab.add(tokens[i])
              self.bigrams.add((tokens[i-1], tokens[i]))


    def calculate_bigram_probability(self, word1, word2):
      if word1 in self.vocab and word2 in self.vocab:
          count_bigram = self.bigram_counts[word1 , word2]
          count_unigram = self.unigram_counts[word1]
          if count_unigram > 0:
              return count_bigram / count_unigram
      return 0.0

    # def calculate_all_bigrams_probabilities(self,bigrams,bigram_counts,unigram_counts):
    #   bigramProbabilities = defaultdict(float)
    #   for bigram in bigrams:
    #       word1 = bigram[0]
    #       word2 = bigram[1]
    #       bigramProbabilities[bigram] = (self.bigram_counts[word1 , word2])/(self.unigram_counts[word1])
    #   return bigramProbabilities

    def calculate_all_bigrams_probabilities(self, corpus):
        bigram_probabilities = {}
        for sentence in corpus:
            tokens = sentence.split()
            for i in range(1, len(tokens)):
                bigram = (tokens[i-1], tokens[i])
                probability = self.calculate_bigram_probability(bigram[0], bigram[1])
                bigram_probabilities[bigram] = probability
        return bigram_probabilities

    def bigram_probability_recursive(self, words):
        if len(words) < 2:
            return 0.0
        word1, word2 = words[0], words[1]
        if word1 in self.vocab and word2 in self.vocab:
            count_bigram = self.bigram_counts[word1, word2]
            count_unigram = self.unigram_counts[word1]
            if count_unigram > 0:
                recursive_probability = self.bigram_probability_recursive(words[1:])
                return count_bigram / count_unigram + recursive_probability
        return 0.0

    def laplace_smoothing(self, words, alpha=1):
        if len(words) < 2:
            return 0.0
        word1, word2 = words[0], words[1]
        count_bigram = self.bigram_counts[word1, word2] + alpha
        count_unigram = self.unigram_counts[word1] + alpha * len(self.vocab)
        recursive_probability = self.laplace_smoothing(words[1:], alpha)
        return count_bigram / count_unigram + recursive_probability

    def calculate_all_laplace_probabilities(self, corpus, alpha=1):
        laplace_probabilities = {}
        for sentence in corpus:
            tokens = sentence.split()
            for i in range(1, len(tokens)):
                bigram = (tokens[i-1], tokens[i])
                probability = self.laplace_smoothing(bigram, alpha)
                laplace_probabilities[bigram] = probability
        return laplace_probabilities

    def calculate_all_kneser_ney_probabilities(self, corpus, d=0.75):
        kneser_ney_probabilities = {}
        for sentence in corpus:
            tokens = sentence.split()
            for i in range(1, len(tokens)):
                bigram = (tokens[i-1], tokens[i])
                probability = self.kneser_ney_smoothing(bigram, d)
                kneser_ney_probabilities[bigram] = probability
            break
        return kneser_ney_probabilities

    def kneser_ney_smoothing(self, words, d=0.75):
        if len(words) < 2:
            return 0.0
        word1, word2 = words[0], words[1]
        count_bigram = self.bigram_counts[word1, word2]
        count_unigram = self.unigram_counts[word1]
        a_word1 = (d * len(set([w for w in self.vocab if self.bigram_counts[word1, w] > 0])))/count_unigram

        p_continuation_word2 = (len(set([w for w in self.vocab if self.bigram_counts[word1, w] > 0])) /
                                len(set([bigram for bigram, count in self.bigram_counts.items() if count > 0])))
        kneser_ney_prob = (max(count_bigram - d, 0) / count_unigram) + a_word1 * p_continuation_word2
        return kneser_ney_prob



    def calculate_bigram_probability_with_emotion(self, word1, word2):
      min_probability = 0.0
      max_probability = 1.0
      base_probability = self.calculate_bigram_probability(word1, word2)
      emotion_scores = self.get_emotion_score(word2)
      print(emotion_scores)
      if emotion_scores is not None:
          beta = max(emotion_scores[0], key=lambda x: x['score'])
          score = beta['score']
          emotion = beta['label']
        #   print("beta",beta)
          normalized_modified_probability = min(max_probability, max(min_probability, base_probability + score))
          return normalized_modified_probability, emotion
      else:
          return base_probability

    def calculate_all_bigram_probabilities_with_emotion(self, corpus):
        bigram_probabilities_with_emotion = {}
        for bigram in self.bigrams:
            probability_array, emotion = self.calculate_bigram_probability_with_emotion(bigram[0], bigram[1])
            bigram_probabilities_with_emotion[bigram] = probability_array, emotion
        return bigram_probabilities_with_emotion

    def bigram_probability_with_emotion(self, words):
        if len(bigram) < 2:
            return 0.0
        word1, word2 = words[0], words[1]
        base_probability = self.calculate_bigram_probability(word1, word2)
        emotion_scores1 = self.get_emotion_score(word1)
        emotion_scores2 = self.get_emotion_score(word2)
        emotion_array = [(score1 + score2) / 2 + base_probability
                             for score1, score2 in zip(emotion_scores1, emotion_scores2)]
        return emotion_array

    def get_emotion_score(self, word):
        emotion_scores = classifier(word)
        return emotion_scores
    
corpus = load_file(filename)

#top5 bigram probabilities
bigram_model = BigramLM()
bigram_model.learn_model(corpus)
all_probabilities = bigram_model.calculate_all_bigrams_probabilities(corpus)

top_5_bigrams = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Bigrams with Probabilities:")
for bigram, probability in top_5_bigrams:
    print(f"{bigram}: {probability}")

#top5 laplace
bigram_model = BigramLM()
bigram_model.learn_model(corpus)
all_probabilities = bigram_model.calculate_all_laplace_probabilities(corpus)
top_5_bigrams = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Bigrams with Probabilities:")
for bigram, probability in top_5_bigrams:
    print(f"{bigram}: {probability}")

#top5 kneser_ney
bigram_model = BigramLM()
bigram_model.learn_model(corpus)
all_probabilities = bigram_model.calculate_all_kneser_ney_probabilities(corpus)
top_5_bigrams = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Bigrams with Probabilities:")
for bigram, probability in top_5_bigrams:
    print(f"{bigram}: {probability}")