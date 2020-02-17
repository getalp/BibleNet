#!usr/bin/env python
#-*- coding: utf8 -*-

# Author of Original Implementation: David Harwath
# https://github.com/dharwath/DAVEnet-pytorch

#
#   BibleNet
#
#   Modification by: GETALP TEAM
#   Last Modified: 27/03/2019
#
#   Université Grenoble Alpes

import math
import pickle
import numpy as np
import torch
import numpy
from pprint import pprint


#
#   DISTANCES
#

def compute_constrastive(distances_matrix, margin):
    # Author of Original Implementation: G. Chrupala
    # https://github.com/gchrupala/vgs

    def diag(M):
        """Return the diagonal of the matrix."""
        I = torch.autograd.Variable(torch.eye(M.size(0))).cuda()
        return (M * I).sum(dim=0)

    diagonal = diag(distances_matrix)
    cost_s = torch.clamp(margin - distances_matrix + diagonal, min=0)
    cost_i = torch.clamp(margin - distances_matrix + diagonal.view(-1, 1), min=0)
    cost_tot = cost_s + cost_i
    I = torch.autograd.Variable(torch.eye(cost_tot.size(0)), requires_grad=True).cuda()
    cost_tot = (1 - I) * cost_tot
    return cost_tot.mean()


def calc_loss(audioA_output, audioB_output, margin=.2, distance='cosine'):
    return compute_constrastive(compute_similarity_matrix(audioA_output, audioB_output, distance), margin)


def compute_similarity_matrix(U, V, distance='cosine'):
    # https://github.com/gchrupala/vgs/blob/master/onion/loss.py
    # Returns the matrix of cosine similarity between each row of U and each row of V.
    if distance != 'cosine':
        raise NotImplementedError

    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return torch.matmul(U_norm, V_norm.t())
#
#   LEARNING & PROGRESS
#

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

#
#   SCORE
#

def calc_scores(similarity_matrix, r_at_n=(1, 5, 10)):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""

    # Code borrowed from G. Chrupala
    # https://github.com/gchrupala/visually-grounded-speech/blob/master/imaginet/evaluate.py

    S = similarity_matrix.cpu().numpy()
    n = S.shape[0]

    # /!\ We assume that matching pairs are along the diagonal
    correct = numpy.fromfunction(lambda i, j: i==j,shape=(n, n), dtype=int)

    correspondances = [] # store query, first retrieve result index and rank of true result
    result = {'ranks': [], 'precision': {}, 'recall': {}, 'overlap': {}}
    for n in r_at_n:
        result['precision'][n] = []
        result['recall'][n] = []
        result['overlap'][n] = []
    
    for j, row in enumerate(S):
        ranked = numpy.argsort(row)
        id_correct = numpy.where(correct[j][ranked])[0]
        rank1 = id_correct[0]+1
        correspondances.append((j, ranked[0], rank1))
        for n in r_at_n:
            id_topn = ranked[:n]
            overlap = len(set(id_topn).intersection(set(ranked[id_correct])))
            
            result['precision'][n].append(overlap / n)
            result['recall'][n].append(overlap / len(id_correct))
            result['overlap'][n].append(overlap)
        result['ranks'].append(rank1)
    
    return result, correspondances

def make_scores(audioA_output, audioB_output, distance='cosine', r_at_n=(1, 5, 10)):
    similarity_matrix = compute_similarity_matrix(audioA_output, audioB_output, distance=distance)
    results, correspondances = calc_scores(similarity_matrix, r_at_n)
    return results, correspondances

def scores(data, r_at_n=(1, 5, 10)):
    return [np.mean(data['recall'][r_at_n[i]]) for i in range(len(r_at_n))] + [np.median(data['ranks'])]


def print_scores(results, r_at_n=(1, 5, 10)):
    print('\t'.join(['r@{}'.format(r) for r in r_at_n]+['rank']))
    print('\t'.join(['{:.3f}'.format(r) for r in scores(results, r_at_n)]))

#
#   OTHER
#

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
