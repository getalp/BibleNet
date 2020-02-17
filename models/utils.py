import numpy as np
import torch

#   BibleNet
#
#   Author: GETALP TEAM
#   Last Modified: 27/03/2019
#
#   Université Grenoble Alpes

class Padder:
    def __init__(self, seq_reduction=None, remove_one=False):
        self.seq_reduction = seq_reduction
        self.remove_one = remove_one

    def __call__(self, input_seq):

        def pad(seq_input):
            max_len = max([len(inp_) for inp_ in seq_input])

            if self.seq_reduction is not None:
                # seq_len mod stride (sum of strides for all convolutional layers) > 0
                # add (difference to get a multiple of stride) - 1
                if max_len % self.seq_reduction > 0:
                    diff = self.seq_reduction - max_len % self.seq_reduction
                    max_len += diff
                    if self.remove_one:
                        max_len -= 1

                # seq_len is a multiple of stride
                # add stride - 1
                elif max_len % self.seq_reduction == 0:
                    max_len += self.seq_reduction
                    if self.remove_one:
                        max_len -= 1
            else:
                max_len = max_len

            seq = [np.pad(inp, [(0, max_len - len(inp)), (0, 0)], mode='constant', constant_values=0) for inp in seq_input]
            seq = torch.FloatTensor(np.array(seq))
            return seq

        audioA = pad([audioA for audioA, _, _, _ in input_seq])
        audioB = pad([audioB for _, audioB, _, _ in input_seq])
        textA  = [textA for _, _, textA, _ in input_seq]
        textB  = [textA for _, _, _, textB in input_seq]

        return audioA, audioB, textA, textB
