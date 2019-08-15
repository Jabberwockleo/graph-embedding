#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : demo_deep_walk.py
# Author            : Wan Li
# Date              : 15.08.2019
# Last Modified Date: 15.08.2019
# Last Modified By  : Wan Li
#
# Deep walk demo (p=q=1)

# Process data
FN_ITEM_TITLE = "../data/item_title.index"
FN_UI_ADJ = "../data/user_items.adjlist"
FN_CORPUS = "../data/random_walks.corpus"

def load_item_titles(fn):
    """
        Load item titles
    """
    item_title_dict = {}
    idx = 0
    with open(fn) as fd:
        for line in fd:
            line = line.rstrip()
            item_title_dict["I{}".format(idx)] = line
            idx += 1
    return item_title_dict

def readable_results(entries, item_title_dict):
    """
        Print readable search results
    """
    for entry in entries:
        if entry[0] in item_title_dict:
            print("{} {} {}".format(entry[0], entry[1], item_title_dict[entry[0]]))
        else:
            print("{} {}".format(entry[0], entry[1]))

item_title_dict = load_item_titles(FN_ITEM_TITLE)
print("I0: {}".format(item_title_dict["I1"]))

def load_user_items_dict(fn):
    """
        Load user-items adjacent dict
    """
    user_items_dict = {}
    with open(fn) as fd:
        for line in fd:
            arr = line.rstrip().split(" ")
            if len(arr) < 2:
                continue
            user_items_dict[arr[0]] = arr[1:]
    return user_items_dict

def readable_user_items(user, user_items_dict, item_title_dict):
    """
        Print readable user's item history
    """
    for item in user_items_dict[user]:
        print("{} {}".format(item, item_title_dict[item]))

user_items_dict = load_user_items_dict(FN_UI_ADJ)
print("U1->items: {}".format(user_items_dict["U1"]))
readable_user_items("U1", user_items_dict=user_items_dict, item_title_dict=item_title_dict)

# Create graph
import networkx as nx
def create_unweighted_graph_with_adjlist_format(fn, is_directed=False):
    '''
        Creates an undirected unweighted graph with adjacent list format file
    '''
    G = nx.read_adjlist(fn, comments='#', delimiter=" ", nodetype=str)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    if is_directed:
        G = G.to_directed()
    return G
G = create_unweighted_graph_with_adjlist_format(FN_UI_ADJ, is_directed=False)

import random
random.sample(list(G.neighbors("U1")), 1)

# Create walker
import graphwalker
walker = graphwalker.GraphWalker(G, p=1, q=1)

def sample_walks(iterator, num_samples):
    arr = []
    for _ in range(num_samples):
        item = next(iterator, None)
        if item is None:
            return arr
        arr.append(item)
    return arr

samples = sample_walks(walker.simulate_walks(num_epochs=1, walk_len=10), num_samples=5)
print(samples)

# Generate corpus
def generate_corpus(generator, fn_corpus):
    """
        Generate corpus
    """
    with open(fn_corpus, "w") as fd:
        for walk in generator:
            fd.write("{}\n".format(" ".join(walk)))
generate_corpus(walker.simulate_walks(num_epochs=5, walk_len=10), FN_CORPUS)

# Learn embeddings
import word2vec as w2v
model = w2v.learn(FN_CORPUS, num_dims=32, window_size=5)

# Save
w2v.save(model, "../output/user_item.emb", "../output/user_item.vocab")

# Load
wv = w2v.load("../output/user_item.emb")

# test
query = "I2"
entries = w2v.search(model.wv, [query], topk=20)
print("IN: {} {}".format(query, item_title_dict[query]))
readable_results(entries, item_title_dict=item_title_dict)
readable_user_items("U82391", user_items_dict=user_items_dict, item_title_dict=item_title_dict)