#! /usr/bin/python
# -*- coding: utf-8 -*-


"""Rank sentences based on cosine similarity and a query."""


from argparse import ArgumentParser
import numpy as np
import operator
import collections

#constants
STOP_WORDS = list("")
#STOP_WORDS = list(".,:;!?-#\'\"\\")

def get_sentences(file_path):
    """Return a list of sentences from a file."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().splitlines()


def get_top_k_words(sentences, k):
    """Return the k most frequent words as a list."""
    # create dict of unique words to count them down
    un_words = {}
    for x in sentences:
        for word in x.split(' '):
            if word not in STOP_WORDS:
                if not un_words.__contains__(word):
                    un_words[word] = 1
                else:
                    un_words[word] += 1
    un_words = sorted(un_words.items(), key=operator.itemgetter(1), reverse=True)
    top_k_words = []
    for x in un_words[0:k]:     # convert list into an array of words only
        top_k_words.append(x[0])
    return top_k_words



def encode(sentence, vocabulary):
    """Return a vector encoding the sentence."""
    temp_vect = []

    c = collections.Counter(sentence.split(" "))
    for word in vocabulary:
        if word in c:
              temp_vect.append(c[word])
        else:
            temp_vect.append(0)
    vect = np.asarray(temp_vect)
    return vect


def get_top_l_sentences(sentences, query, vocabulary, l):
    u = encode(query, vocabulary)
    sim = {}
    for x in sentences:
        sim[x]=cosine_sim(u,encode(x,vocabulary))
    sim = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
    res_list = sim[0:l]
    """
    For every sentence in "sentences", calculate the similarity to the query.
    Sort the sentences by their similarities to the query.

    Return the top-l most similar sentences as a list of tuples of the form
    (similarity, sentence).
    """
    return res_list


def cosine_sim(u, v):
    """Return the cosine similarity of u and v."""
    sim = np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return sim


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='An input file containing sentences, one per line')
    arg_parser.add_argument('QUERY', help='The query sentence')
    arg_parser.add_argument('-k', type=int, default=1000,
                            help='How many of the most frequent words to consider')
    arg_parser.add_argument('-l', type=int, default=10, help='How many sentences to return')
    args = arg_parser.parse_args()

    sentences = get_sentences(args.INPUT_FILE) # records all sentences file consists of
    top_k_words = get_top_k_words(sentences, args.k)
    query = args.QUERY.lower()
    print (query)

    # suppress numpy's "divide by 0" warning.
    # this is fine since we consider a zero-vector to be dissimilar to other vectors
    with np.errstate(invalid='ignore'):
        result = get_top_l_sentences(sentences, query, top_k_words, args.l)

    print('result:')
    for sentence, sim in result:
        print("{:.5f}\t{}".format(sim, sentence))

if __name__ == '__main__':
    main()
