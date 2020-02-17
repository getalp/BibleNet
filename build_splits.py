#!usr/bin/env python
#-*- coding: utf8 -*-

#   BibleNet
#
#   Author: GETALP TEAM
#   Last Modified: 27/03/2019
#
#   Université Grenoble Alpes

import sys
import numpy as np
import json
import os
from pprint import pprint

if len(sys.argv)>=10:
    seed = int(sys.argv[9])
    np.random.seed(seed)
else:
    seed = 482
    np.random.seed(482)

def load_csv(csv_name):
    with open(csv_name, 'r') as csv_file:
        csv_lines = [line.strip().split(',') for line in csv_file]

    lang_list = csv_lines[0]
    verse_list = np.array(csv_lines[4:])
    lang_path = csv_lines[1]
    return lang_list, verse_list, lang_path

def build_train_val_test(data, train_perc, val_perc, test_perc):
    if train_perc + val_perc + test_perc != 1:
        exit('Error: Splits percentage should sum up to 1')

    train_end = int(train_perc * len(data))
    val_end = train_end + int(val_perc * len(data)) 

    train, val, test = np.array(data[0:train_end]), np.array(data[train_end:val_end]), np.array(data[val_end:])
    return train, val, test

def main(csv_name, lang_a, lang_b, train_perc, val_perc, test_perc, common_verses=True, shuffle_me=True):
    lang_list, verse_list, lang_path = load_csv(csv_name)

    if (lang_a not in lang_list[1:]) or (lang_b not in lang_list[1:]):
        exit('Error: {} or {} not an available language. (List of languages: {})'.format(lang_a, lang_b, ', '.join(lang_list[1:])))

    if shuffle_me:
        np.random.shuffle(verse_list)


    if common_verses:
        available_ids = [line for line in range(len(verse_list)) 
                        if 'Not Available' not in verse_list[line, :]]
    else:
        available_ids = [line for line in range(len(verse_list)) 
                        if (verse_list[line, lang_list.index(lang_a)]!='Not Available' and verse_list[line, lang_list.index(lang_b)]!='Not Available')]
    print('Available verses: {}'.format(len(available_ids)))

    train_ids, val_ids, test_ids = build_train_val_test(available_ids, train_perc, val_perc, test_perc)

    splits = {'train': verse_list[train_ids[:,None], [0,lang_list.index(lang_a),lang_list.index(lang_b)]],
              'val': verse_list[val_ids[:,None], [0,lang_list.index(lang_a),lang_list.index(lang_b)]],
              'test':verse_list[test_ids[:,None], [0,lang_list.index(lang_a),lang_list.index(lang_b)]]}

    for split in sorted(splits.keys()):
        print('Building split {}'.format(split))
        json_data = {
            'audio_langA_base_path':lang_path[lang_list.index(lang_a)],
            'audio_langB_base_path':lang_path[lang_list.index(lang_b)],
            'data':[]
        }

        c=0
        for verse_id, verse_langA, verse_langB in splits[split]:
            c+=1
            print('[{}/{}] {} {} {}'.format(c, len(splits[split]), verse_id, verse_langA, verse_langB))
            with open(os.path.join(lang_path[lang_list.index(lang_a)].replace('/wav/', '/txt/'), verse_langA+'.txt'), encoding='utf8') as lang_a_text_file:
                lang_a_text=lang_a_text_file.read().strip()
            
            with open(os.path.join(lang_path[lang_list.index(lang_b)].replace('/wav/', '/txt/'), verse_langB+'.txt'), encoding='utf8') as lang_b_text_file:
                lang_b_text=lang_b_text_file.read().strip()

            verse_dict = {
                'uttid':int(verse_id),
                'text-langA':lang_a_text,
                'text-langB':lang_b_text,
                'wav-langA':verse_langA+'.wav',
                'wav-langB':verse_langB+'.wav'
            }

            json_data['data'].append(verse_dict)

        outdir = './data/{}-{}'.format(lang_a, lang_b)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(os.path.join(outdir, '{}-{}-{}-seed{}.json'.format(split, lang_a, lang_b, seed)), 'w') as dump_json_file:
            print('Dumping {}'.format(split))
            json.dump(json_data, dump_json_file)


if __name__ == '__main__':
    
    if len(sys.argv) < 8:
        print(len(sys.argv))
        print('Usage: <str csv_filename> <str lang_a> <str lang_b> <bool common_verses_to_all_languages_only> <bool shuffle> <float train %> <float val %> <float test %> [<int seed>]')
        exit()

    csv_name = sys.argv[1]
    lang_a = sys.argv[2]
    lang_b = sys.argv[3]
    common_verses = sys.argv[4].lower() == 'true'
    shuffle_me = sys.argv[5].lower() == 'true'
    args_train_val_test = sys.argv[6:9]
    print(args_train_val_test)
    train_perc, val_perc, test_perc = map(float, args_train_val_test)


    print('\tCSV Filename: {}\n\
           \n\tLang A: {}\
           \n\tLang B: {}\n\
           \n\tTrain %: {}\
           \n\tVal %: {}\
           \n\tTest %: {}\n\
           \n\tCommon verses to all languages only: {}\n\
           \n\tShuffle: {}\
           \n\tSeed: {}'.format(csv_name, lang_a, lang_b, train_perc, val_perc, test_perc, common_verses, shuffle_me, seed))

    main(csv_name, lang_a, lang_b, train_perc, val_perc, test_perc, common_verses=common_verses, shuffle_me=shuffle_me)    
