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

import argparse
import os
import pickle
import sys
import time
import torch

import dataloaders
import models
from steps import train_bible, validate_bible
from models.utils import Padder

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='',
        help="training data json")
parser.add_argument("--data-val", type=str, default='',
        help="validation data json")
parser.add_argument("--exp-dir", type=str, default="",
        help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="sgd",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=100, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=25,
        help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100,
        help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="BibleNet",
        help="audio model architecture", choices=["BibleNet"])
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")

args = parser.parse_args()

resume = args.resume

if args.resume:
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume
        
print(args)

collate_fn = Padder()

train_loader = torch.utils.data.DataLoader(
    dataloaders.AudioBibleDataset(args.data_train),
    batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloaders.AudioBibleDataset(args.data_val),
    batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

audio_model_langA = models.BibleNet()
audio_model_langB = models.BibleNet()

if not bool(args.exp_dir):
    print("exp_dir not specified, automatically creating one...")
    lang_a_name, lang_b_name = os.path.basename(args.data_train).split('-')[1:3]
    args.exp_dir = "exp/Model-%s-%s/pid=%s_server=%s_Optim=%s_Margin=%s_LR=%s_Epochs=%s_Time=%s" % (
        lang_a_name, lang_b_name, os.getpid(), os.uname()[1], args.optim, args.margin, args.lr, args.n_epochs, time.asctime().replace(' ', '-').replace(':', '-'))

if not args.resume:
    print("\nexp_dir: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

train_bible(audio_model_langA, audio_model_langB, train_loader, val_loader, args)
