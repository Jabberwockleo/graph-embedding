#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : aliasmethod.py
# Author            : Wan Li
# Date              : 15.08.2019
# Last Modified Date: 15.08.2019
# Last Modified By  : Wan Li
#
# Alias method for sampling
# Reference:
#     R. A. Kronmal and A. V. Peterson.
#     On the alias method for generating random variables
#     from a discrete distribution. The American Statistician,
#     33(4):214-218, 1979.
# https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

import numpy as np
import numpy.random as npr

def alias_setup(probs):
    """
        Setup alias sample bins
        Params:
            probs: prob for each category
        Return:
            J: dict[category] conjugate category dict
            q: dict[category] threshold dict for choosing the conjugate category
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
        Sample from alias bins
        Params:
            J: dict[category] conjugate category dict
            q: dict[category] threshold dict for choosing the conjugate category
        Return:
            categorical index
    """
    K  = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]

if __name__ == "__main__":
    K = 5
    N = 1000

    # Get a random probability vector.
    probs = npr.dirichlet(np.ones(K), 1).ravel()

    # Construct the table.
    J, q = alias_setup(probs)

    # Generate variates.
    X = np.zeros(N)
    for nn in range(N):
        X[nn] = alias_draw(J, q)
    print(X)