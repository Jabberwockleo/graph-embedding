#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : graphwalker.py
# Author            : Wan Li
# Date              : 15.08.2019
# Last Modified Date: 15.08.2019
# Last Modified By  : Wan Li

import importlib as imp
import random
import aliasmethod as am
am = imp.reload(am)

class GraphWalker(object):
    """
        Wraps a NetworkX graph and associates some random walk methods
    """
    def __init__(self, G, p, q, is_weighted=True, num_sample_neighbors=None):
        """
            Initializer
            Params:
                G: a NetworkX graph instance
                p: the return hyperparameter
                q: the inout hyperparameter
                is_weighted: whether edges are weighted differently
                num_sample_neighbors: if not None (by default)
                    sample num_sample_neighbors neighbors
                    if number of neighbors is greater than num_sample_neighbors
                The unnormalized walk probability is computed as follows:
                    W(preceding node's neigbors) = 1
                    W(preceding node) = 1/p
                    W(outward farther nodes) = 1/q
                (Reference: [Grover, Leskovec, KDD 2016])
        """
        self.G = G
        self.p = p
        self.q = q
        self.is_weighted = is_weighted
        self.num_sample_neighbors = num_sample_neighbors
        if p == 1 and q == 1 and is_weighted == False:
            self.is_vanilla_deep_walk = True
        else:
            self.is_vanilla_deep_walk = False
            self.__precompute_transition_probabilities()


    def __uniform_sample(arr, num):
        """
            Uniformly sample from list
            Params:
                arr: list to sample from
            Return:
                Sampled list
        """
        sampled = []
        interval = 1.0 / (num - 1)
        for intidx in range(num):
            idx = math.floor(intidx * interval)
            sampled.append(arr[idx])
        return sampled


    def __setup_alias_node2node(self, node):
        """
            Setup alias bins for an 'current node' -> 'next node'
            Params:
                src: start node of current edge
                dst: end node of current edge
        """
        G = self.G
        adjacent_nodes = sorted(G.neighbors(node))
        if self.num_sample_neighbors is not None:
            if len(adjacent_nodes) > self.num_sample_neighbors:
                adjacent_nodes = self.__uniform_sample(adjacent_nodes, self.num_sample_neighbors)

        unnormalized_probs = [G[node][nbr]['weight'] for nbr in adjacent_nodes]
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return am.alias_setup(normalized_probs)


    def __setup_alias_edge2node(self, src, dst):
        """
            Setup alias bins for an 'current edge' -> 'next node'
                note: current edge[0]=src is previous node
                      current edge[1]=dst is current node
            Params:
                src: start node of current edge
                dst: end node of current edge
        """
        G = self.G
        p = self.p
        q = self.q

        adjacent_nodes = sorted(G.neighbors(dst)) # sorting ensures consistency
        if self.num_sample_neighbors is not None:
            if len(adjacent_nodes) > self.num_sample_neighbors:
                adjacent_nodes = self.__uniform_sample(adjacent_nodes, self.num_sample_neighbors)

        unnormalized_probs = []
        for dst_nbr in adjacent_nodes:
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return am.alias_setup(normalized_probs)
            

    def __precompute_transition_probabilities(self):
        """
            Precompute transition probabilities to reduce random walk overhead
        """
        G = self.G
        is_directed = G.is_directed()
        
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        
        alias_nodes = {}
        cnt = 0
        for node in G.nodes():
            alias_nodes[node] = self.__setup_alias_node2node(node)
            cnt += 1
            if cnt % 10000 == 0:
                print("processed {}/{} nodes.".format(cnt, num_nodes))
        self.alias_nodes = alias_nodes
        
        alias_edges = {}
        if is_directed:
            cnt = 0
            for edge in G.edges():
                alias_edges[(edge[0], edge[1])] = self.__setup_alias_edge2node(edge[0], edge[1])
                cnt += 1
                if cnt % 10000 == 0:
                    print("processed {}/{} edges.".format(cnt, num_edges))
        else:
            cnt = 0
            for edge in G.edges():
                alias_edges[(edge[0], edge[1])] = self.__setup_alias_edge2node(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.__setup_alias_edge2node(edge[1], edge[0])
                cnt += 1
                if cnt % 10000 == 0:
                    print("processed {}/{} edges.".format(cnt, num_edges))
        self.alias_edges = alias_edges


    def generate_one_walk(self, start_node, walk_len):
        """
            Generate one walk
        """
        G = self.G
        walk = [start_node]
        while len(walk) < walk_len:
            cur_node = walk[-1]
            if self.is_vanilla_deep_walk == True:
                next_node = random.sample(list(self.G.neighbors(cur_node)), 1)[0]
                walk.append(next_node)
            else:
                cur_nbrs = sorted(G.neighbors(cur_node))
                if len(cur_nbrs) > 0:
                    if len(walk) == 1:
                        next_node = cur_nbrs[am.alias_draw(
                            self.alias_nodes[cur_node][0],
                            self.alias_nodes[cur_node][1])]
                        walk.append(next_node)
                    else:
                        prev_node = walk[-2]
                        next_node = cur_nbrs[am.alias_draw(
                            self.alias_edges[(prev_node, cur_node)][0], 
                            self.alias_edges[(prev_node, cur_node)][1])]
                        walk.append(next_node)
                else:
                    break
        return walk


    def simulate_walks(self, num_epochs, walk_len):
        """
            Generator of random walks
            Params:
                num_epochs: number of epochs for every node being the start node
                walk_len: length of each sequence generated by one random walk
        """
        all_nodes = list(self.G.nodes())
        for epoch in range(1, num_epochs + 1):
            print("epoch: {}/{}".format(epoch, num_epochs))
            random.shuffle(all_nodes)
            cnt = 0
            for node in all_nodes:
                yield self.generate_one_walk(node, walk_len)
                cnt += 1
                if cnt % 10000 == 0:
                    print(" processed: {}/{}".format(cnt, len(all_nodes)))
