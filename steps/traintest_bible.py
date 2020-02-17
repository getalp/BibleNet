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

import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .util_bible import *

def train_bible(audio_model_langA, audio_model_langB, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)

    if not isinstance(audio_model_langA, torch.nn.DataParallel):
        audio_model_langA = nn.DataParallel(audio_model_langA)

    if not isinstance(audio_model_langB, torch.nn.DataParallel):
        audio_model_langB = nn.DataParallel(audio_model_langB)

    if epoch != 0:
        audio_model_langA.load_state_dict(torch.load("%s/models/audio_model_langA.%d.pth" % (exp_dir, epoch)))
        audio_model_langB.load_state_dict(torch.load("%s/models/audio_model_langB.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model_langA = audio_model_langA.to(device)
    audio_model_langB = audio_model_langB.to(device)
    
    # Set up the optimizer
    audio_trainables = [p for p in audio_model_langA.parameters() if p.requires_grad]
    image_trainables = [p for p in audio_model_langB.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
    if args.optim == 'sgd':
       optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    print("torch seed {}".format(torch.initial_seed()))

    audio_model_langA.train()
    audio_model_langB.train()
    while True:
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model_langA.train()
        audio_model_langB.train()
        for i, (audioA, audioB, _, _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audioB.size(0)

            audioA_input = audioA.to(device)
            audioB_input = audioB.to(device)

            optimizer.zero_grad()

            audioA_output, alphasA = audio_model_langA(audioA_input)
            audioB_output, alphasB = audio_model_langB(audioB_input)

            loss = calc_loss(audioA_output, audioB_output, margin=args.margin)

            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        result = validate_bible(audio_model_langA, audio_model_langB, test_loader, args)
        
        avg_acc = np.median(result['ranks'])

        torch.save(audio_model_langA.state_dict(),
                "%s/models/audio_model_langA.%d.pth" % (exp_dir, epoch))
        torch.save(audio_model_langB.state_dict(),
                "%s/models/audio_model_langB.%d.pth" % (exp_dir, epoch))
        torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
        
        if avg_acc > best_acc:
            best_epoch = epoch
            best_acc = avg_acc
            shutil.copyfile("%s/models/audio_model_langA.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_audio_model_langA.pth" % (exp_dir))
            shutil.copyfile("%s/models/audio_model_langB.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_audio_model_langB.pth" % (exp_dir))
        _save_progress()
        epoch += 1

def validate_bible(audio_model_langA, audio_model_langB, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model_langA, torch.nn.DataParallel):
        audio_model_langA = nn.DataParallel(audio_model_langA)
    if not isinstance(audio_model_langB, torch.nn.DataParallel):
        audio_model_langB = nn.DataParallel(audio_model_langB)
    audio_model_langA = audio_model_langA.to(device)
    audio_model_langB = audio_model_langB.to(device)
    # switch to evaluate mode
    audio_model_langB.eval()
    audio_model_langA.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    AudioA_embeddings = [] 
    AudioB_embeddings = []
    with torch.no_grad():
        for i, (audioA, audioB, _, _) in enumerate(val_loader):
            audioA_input = audioA.to(device)
            audioB_input = audioB.to(device)

            # compute output
            audioA_output, alphasA = audio_model_langA(audioA_input)
            audioB_output, alphasB = audio_model_langB(audioB_input)

            audioA_output = audioA_output.to('cpu').detach()
            audioB_output = audioB_output.to('cpu').detach()

            AudioA_embeddings.append(audioA_output)
            AudioB_embeddings.append(audioB_output)
            
            batch_time.update(time.time() - end)
            end = time.time()

        audioA_output = torch.cat(AudioA_embeddings)
        audioB_output = torch.cat(AudioB_embeddings)

        result, _ = make_scores(audioA_output, audioB_output)
        print_scores(result)
    return result
