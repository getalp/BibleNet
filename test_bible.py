# Author of Original Implementation: David Harwath
# https://github.com/dharwath/DAVEnet-pytorch

#
#   BibleNet
#
#   Modification by: GETALP TEAM
#   Last Modified: 27/03/2019
#
#   Université Grenoble Alpes

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import dataloaders
import models
from steps.util_bible import *
import json
from models.utils import Padder

DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scores(data, ns_val=(1, 5, 10)):
        return [np.mean(data['recall'][ns_val[i]]) for i in range(len(ns_val))] + [np.median(data['ranks'])]

def encode_test_split(audio_model_langA, audio_model_langB, data_provider):
    audio_model_langB.eval()
    audio_model_langA.eval()

    N_examples = data_provider.dataset.__len__()
    AudioA_embeddings = [] 
    AudioB_embeddings = []
    with torch.no_grad():
        for i, (audioA, audioB, _, _) in enumerate(data_provider):
            audioA_input = audioA.to(DEVICE)
            audioB_input = audioB.to(DEVICE)

            # compute output
            audioA_output, _ = audio_model_langA(audioA_input)
            audioB_output, _ = audio_model_langB(audioB_input)

            audioA_output = audioA_output.to('cpu').detach()
            audioB_output = audioB_output.to('cpu').detach()

            AudioA_embeddings.append(audioA_output)
            AudioB_embeddings.append(audioB_output)

        audioA_output = torch.cat(AudioA_embeddings)
        audioB_output = torch.cat(AudioB_embeddings)

    return audioA_output,audioB_output

def main(args):
    model_path = args[0]
    model_epoch = int(args[1])
    test_path = args[2]

    collate_fn = Padder()
    # Get test data
    test_loader = torch.utils.data.DataLoader(dataloaders.AudioBibleDataset(test_path), batch_size=1, shuffle=False, collate_fn=collate_fn,  num_workers=8, pin_memory=True)
    
    # Instanciate Model
    audio_model_langA = models.BibleNet()
    audio_model_langB = models.BibleNet()

    if not isinstance(audio_model_langA, torch.nn.DataParallel):
        audio_model_langA = nn.DataParallel(audio_model_langA)
    if not isinstance(audio_model_langB, torch.nn.DataParallel):
        audio_model_langB = nn.DataParallel(audio_model_langB)

    # Load Model Weights
    audio_model_langA.load_state_dict(torch.load("%s/models/audio_model_langA.%d.pth" % (model_path, model_epoch)))
    audio_model_langB.load_state_dict(torch.load("%s/models/audio_model_langB.%d.pth" % (model_path, model_epoch)))
    audio_model_langA = audio_model_langA.to(DEVICE)
    audio_model_langB = audio_model_langB.to(DEVICE)

    # Encode verses
    audio_langA_out, audio_langB_out = encode_test_split(audio_model_langA, audio_model_langB, test_loader)

    # Compute similarity
    similarity_matrix = compute_similarity_matrix(audio_langA_out, audio_langB_out)

    # Get score
    results, correspondances = calc_scores(similarity_matrix)
    
    sys.stdout.write('model_epoch{}\t{:.3f}\t{:.3f}\t{:.3f}\t{}\n'.format(model_epoch,*scores(results, (1,5,10))))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit('Usage: <model_path> <model_epoch> <test_corpus>')
    #print(args)
    main(args)
