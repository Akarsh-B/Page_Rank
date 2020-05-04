#!/usr/bin/env python3
'''
__author__ = {'Akarsh Balasubramanyam'}
__UIN__ = {'658639093'}
__email__ = {'abalas26@uic.edu'}
'''
#---Imports---
# Importing Counter from collections to count the number of tokens
from collections import Counter
# NLTK porter stemmer to stem the tokens to its root form
from nltk.stem import PorterStemmer
#To read command line arguments
import sys
#Importing numpy library for replacing characters
import numpy as np
#Importing pandas to store the tfidf values
import pandas as pd
#Importing spacy for NLP
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
#Zip file manipulations(read/write/open/close)
from zipfile import ZipFile
#Networkx package for creating Graphs
import networkx as nx
#Matlotlib for visualizing the Graphs
import matplotlib.pyplot as plt
#NLTK for generating ngrams
import nltk
from nltk.util import ngrams

#--- Initializations ---
#Spacy load English language
nlp = spacy.load('en_core_web_sm')
#List to extract only Nouns and Adjectives
nouns_and_adjectives = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']
#Porter Stermmer Initialization
ps = PorterStemmer()
#Initialize Tokenizer
tokenizer = Tokenizer(nlp.vocab)
window_size = int(sys.argv[2]) #The window size to be used to generate the graph
alpha = 0.85 # The alpha value for the page rank
max_iter = 10 # The Max iterations that the page-rank algorithm must run before returning the values
k = 10 # Top k ranked documents
n_grams_iter = 3 #Generate n_grams_iter n_grams
#--- Initializations ---

#Function to read document from a zip/tar file without extracting it
#Read the complete contents of all the files and store them in a list for further manipulations
def read_documents(filename):
    abstracts = []
    golds = []
    archive = ZipFile(filename, 'r')
    for file in archive.namelist():
        with archive.open(file, 'r') as file_to_read:
            index = file.split('/')[-1]
            if 'abstracts' in file and '__MACOSX' not in file and index not in ['', 'abstract']:
                abstract = file_to_read.read().decode('iso8859-1')
                abstracts.append([int(index), abstract])
            elif 'gold' in file and '__MACOSX' not in file and index not in ['', '.DS_Store']:
                gold = file_to_read.read().decode('iso8859-1')
                golds.append([int(index), gold])
    return abstracts, golds

#Function to Create abstract tokens
def create_tokens_abstracts(sentence):
    sentence_tokenized = tokenizer(sentence.strip())
    tokens = [token for token in sentence_tokenized if not(token.is_punct) and not(token.is_space) and not(token.is_stop)]
    tokens = [ps.stem(token.text.split('_')[0]) for token in tokens if token.text.split('_')[-1] in nouns_and_adjectives]
    return tokens

#Function to create golds tokens
def create_tokens_golds(sentence):
    gold_tokenized = tokenizer(sentence.strip())
    gold_tokens = [ps.stem(gold_token.text).strip() for gold_token in gold_tokenized if not(gold_token.is_punct)]
    gold_tokens = [gold_token for gold_token in gold_tokens if gold_token != '']
    return gold_tokens

#Function to generate a graph of the abstract collections
def generate_graph(data, window_size):
    document = data
    G = nx.Graph()
    G.clear()
    window = []
    document_tokenized = tokenizer(document.strip())
    for token in document_tokenized:
        if not(token.is_punct) and not(token.is_space):
            token_slice = token.text.split('_')
            token_str = token_slice[0]
            token_pos = token_slice[1]
            if token_pos not in nouns_and_adjectives or token.is_stop:
                window = []
                continue
            token = ps.stem(token_str)
            if not G.has_node(token):
                G.add_node(token)
            if window:
                for window_token in window:
                    if G.has_edge(window_token, token):
                        edge_weight = G[window_token][token]['weight']
                        G[window_token][token]['weight'] += 1
                    else:
                        G.add_edge(window_token, token, weight=1)
            window.append(token)
            if len(window) >= window_size:
                window.pop(0)
    return G

#Function to generate a matrix from graph
def generate_matrix(graph):
    return nx.to_numpy_matrix(graph, weight='weight')

#function to return the page-rank for the document
def page_rank(G, alpha, max_iter):
    if G.number_of_nodes == 0:
        s = {}
    if not G.is_directed():
        G = G.to_directed()
    G_norm =  G
    number_of_nodes = G_norm.number_of_nodes()
    s = dict.fromkeys(G_norm, 1/number_of_nodes)
    p_i = dict.fromkeys(G_norm, 1/number_of_nodes)
    for iter in range(max_iter):
        s_prev = s
        s = dict.fromkeys(s_prev.keys(), 0)
        for node in s:
            #print('Node:',node)
            sum = 0
            for word in G_norm:
                #print('Word:',word)
                if G_norm.has_edge(node, word):
                    sum_weights = 0
                    for adj_word in G_norm:
                        if G_norm.has_edge(word, adj_word):
                            #print('Word,adj_word:',word,adj_word)
                            sum_weights += G_norm[word][adj_word]['weight']
                    #print('Total Weights of Adjacent nodes',sum_weights)
                    #print('Weight:',G_norm[node][word]['weight'])
                    sum += (G_norm[node][word]['weight'] /
                            sum_weights) * s_prev[word]
                    #print('Summation:',sum)
                else:
                    continue
            sum = alpha * sum
            #print('First_term:',sum)
            s[node] = sum + ((1 - alpha) * p_i[node])
            #print('Score:',s[node])
        if s == s_prev:
            break
    return s

#Function to generate ngrams for the abstracts
def generate_ngrams(document, n):
    gram_dictionary = {}
    window = []
    document_tokenized = tokenizer(document.strip())
    for token in document_tokenized:
        if not(token.is_punct) and not(token.is_space):
            token_slice = token.text.split('_')
            token_str = token_slice[0]
            token_pos = token_slice[1]
            if token_pos not in nouns_and_adjectives or token.is_stop:
                window = []
                continue
            token = ps.stem(token_str)
            window.append(token)
            if len(window) > n:
                window.pop(0)
            if len(window) == n:
                if n == 1:
                    gram_dictionary[''.join(window)] = 0
                else:
                    gram_dictionary[tuple(window)] = 0
    return gram_dictionary

#Function to score the generated n_grams
def score(data, n):
    for index, row in data.iterrows():
        for words in list(row[str(n) + '_gram'].keys()):
            score = row[str(n) + '_gram'][words]
            if n == 1:
                if words in list(row['abstract_page_rank'].keys()):
                    score += row['abstract_page_rank'][words]
                else:
                    score += 0
            else:
                for word in words:
                    if word in list(row['abstract_page_rank'].keys()):
                        score += row['abstract_page_rank'][word]
                    else:
                        score += 0
            row[str(n) + '_gram'][words] = score
    return

#Function to generate n_grams for the phrases in gold
def generate_gold_ngrams(sentences):
    gold_ngrams = []
    for words in sentences.split('\n'):
        tokens = tokenizer(words.strip())
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = ps.stem(token.text)
            stemmed_tokens.append(stemmed_token)
        if stemmed_tokens != []:
            if len(stemmed_tokens) != 1:
                gold_ngrams.append(tuple(stemmed_tokens))
            else:
                gold_ngrams.append(''.join(stemmed_tokens))
    return gold_ngrams

#Function to combine all the n_grams generated
def generate_abstaract_n_grams(data, n_grams_iter):
    variable = ''
    for i in range(1, n_grams_iter+1):
        variable += "**row['" + str(i) + "_gram'],"
    data['abstract_n_grams'] = ""
    for index, row in data.iterrows():
        row_data = (eval('{' + str(variable[:-1]) + '}'))
        row['abstract_n_grams'] = row_data
    return

#Function to generate the Mean Reciprocal Rank for all documents and each document having top k phrases
def get_mrr(data, k):
    MRR = []
    for index, row in data.iterrows():
        mrr = []
        for k in range(1, k+1):
            #print(k)
            rank = 0
            phrases = Counter(row['abstract_n_grams']).most_common(k)
            for phrase in phrases:
                if phrase[0] in row['gold_n_grams']:
                    #print(phrase,phrases.index(phrase)+1)
                    rank = phrases.index(phrase)+1
                    break
            if rank:
                rank = 1 / rank
            mrr.append(rank)
        MRR.append(mrr)
    return MRR

#Function to generate the over-all MRR
def get_MRR(MRR, data):
    current = np.zeros_like(MRR[0])
    for mrr in MRR:
        final = np.add(current, np.array(mrr))
        current = final
    final = final/data.shape[0]
    return final

def main():
    #Read documents from the zip file: executing - func:read_documents()
    abstracts, golds = read_documents(sys.argv[1])
    print('Gathering the data...')
    #Add the sentences to Dataframe and keep sentences that are both in abstracts and golds
    abstracts_data = pd.DataFrame(data=abstracts, columns=['index', 'abstracts'])
    golds_data = pd.DataFrame(data=golds, columns=['index', 'golds'])
    www_data = abstracts_data.merge(golds_data, how='inner', on='index').drop('index', axis=1)

    #Apply the function to the respective rows
    www_data['abstract_tokens'] = www_data['abstracts'].apply(create_tokens_abstracts)
    www_data['gold_tokens'] = www_data['golds'].apply(create_tokens_golds)

    print('Generating Graph and Matrix...')

    www_data['abstract_graph'] = www_data['abstracts'].apply(generate_graph, args=(window_size,))
    www_data['abstract_martix'] = www_data['abstract_graph'].apply(generate_matrix)

    print('Calculating Page Rank...')

    #Generate Page rank for all the documents in the dataframe
    www_data['abstract_page_rank'] = www_data['abstract_graph'].apply(page_rank, args=(alpha, max_iter,))

    print('Generating n_grams...')

    #Generate the n_grams for each abstracts and score them based on the page ranks
    for i in range(1, n_grams_iter+1):
        www_data[str(i) + '_gram'] = www_data['abstracts'].apply(generate_ngrams, args=(i,))
        score(www_data, i)

    #Generate n_grams for the gold phrases and combine all the scored abstract n_grams
    www_data['gold_n_grams'] = www_data['golds'].apply(generate_gold_ngrams)
    generate_abstaract_n_grams(www_data, n_grams_iter)

    print('Calculating MRR')

    #Get the MRR for top k ngrams for all the documents combined with a given window size
    MRR = get_mrr(www_data, k)
    mean_MRR = get_MRR(MRR, www_data)
    for i in range(k):
        print('MRR for top {} n_grams with window size {} : {}'.format(i+1, window_size, mean_MRR[i]))

if __name__ == '__main__':
    main()
