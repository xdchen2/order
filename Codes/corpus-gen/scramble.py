# permute_json sentences with 1-gram
# superGlue
# Three main funcs
# 1 permute_json sents
# 2 compute distance between permutation and original sents (control acc)
# 3 save the computed score to the orginal json file where the original sents were extracted

import os
import re
import pandas as pd
import random
import json
import string
from textdistance import levenshtein as ld
from para import params

#################
#     Utils     #
#################


def shuffle_sents(sent):
    ''' input a single sent list
    out: list of rnd
    '''
    random.shuffle(sent)

    return sent


def score(sent1, sent2):
    ''' score two strings with levenshtein distance
    '''
    norm_score = ld.normalized_similarity(sent1, sent2)
    return norm_score


# main fun 1: score rnds and the original ones
def control_acc(json_permu, json_orgin, task):
    '''read in two json
    add a score to each item
    scores are averaged over all sents in each item
    '''

    # read in two json files
    with open(json_permu, 'r') as f:
        data_permu = [json.loads(line) for line in f]
    with open(json_orgin, 'r') as f:
        data_orgin = [json.loads(line) for line in f]

    # multirc is special as it has embeded items
    tasks = params['TASK'][task]
    item_in_json = []
    for pline, oline in zip(data_permu, data_orgin):  # loop over sents in data
        item_score = []
        for v in tasks:  # loop over items needs to be permuted
            osent = oline[v].strip().strip('.').split(
                '.') if task != 'multirc' else oline[v]['text'].strip().strip('.').split('.')
            psent = pline[v].strip().strip('.').split(
                '.') if task != 'multirc' else pline[v]['text'].strip().strip('.').split('.')
            for pss, oss in zip(psent, osent):
                pss = re.sub(r' +', ' ', pss)
                oss = re.sub(r' +', ' ', oss)
                sent_score = score(pss.strip(), oss.strip())
                item_score.append(sent_score)
        pline['score'] = sum(item_score) / len(item_score)

        item_in_json.append(pline)

    return item_in_json


# main fun 2: permutate sentences
def permute_json(jsonf, task):
    ''' Take in a json
    return a json
    edit the value
    '''

    # read in a json
    with open(jsonf, 'r') as f:
        data = [json.loads(line) for line in f]

    #
    task_content_list = []

    tasks = params['TASK'][task]
    for line in data:  # loop over sents in data
        for v in tasks:  # loop over items needs to be permuted

            sentence = line[v].strip().split(
                '.') if task != 'multirc' else line[v]['text'].strip().split('.')
            edited = [' '.join(shuffle_sents(sent.split(' ')))
                      for sent in sentence]

            if task != 'multirc':
                line[v] = '. '.join(edited)
            else:
                line[v]['text'] = '. '.join(edited)

        task_content_list.append(line)

    return task_content_list


# main fun 2.1: permutate sentences
def bimute_json(jsonf, task):
    ''' 
    jsonf: json file containing pred, texts
    return a json
    '''

    # read in a json
    with open(jsonf, 'r') as f:
        data = [json.loads(line) for line in f]
    #
    task_content_list = []
    tasks = params['TASK'][task]
    for line in data:  # loop over sents in data
        for v in tasks:  # loop over items needs to be permuted

            sentence = line[v].strip().split(
                '.') if task != 'multirc' else line[v]['text'].strip().split('.')
            edited = [' '.join(shuffle_sents(sent.split(' ')))
                      for sent in sentence]
            orginl = [' '.join(sent.split(' '))
                      for sent in sentence]

            if task != 'multirc':
                line[v] = '. '.join(edited) + ' <s> ' + '. '.join(orginl)
            else:
                line[v]['text'] = '. '.join(edited) + ' <s> ' + '. '.join(orginl)

        task_content_list.append(line)

    return task_content_list


# main fun 3: create a csv file with the header: org, rnd, inum, keys
def create_pmt(json_permu, json_orgin, task):
    '''read in two json
    read in two versions of sents
    create a csv file
    '''

    # read in two json files
    with open(json_permu, 'r') as f:
        data_permu = [json.loads(line) for line in f]
    with open(json_orgin, 'r') as f:
        data_orgin = [json.loads(line) for line in f]

    # multirc is special as it has embeded items
    tasks = params['TASK'][task]
    item_in_csv = ['org' + '|' +
                   'rnd' + '|' + 'inum' + '|' + 'keys']
    item = 1
    for pline, oline in zip(data_permu, data_orgin):  # loop over each items in data
        for v in tasks:  # loop over sents needs to be permuted
            osent = oline[v].strip().strip(
                '.') if task != 'multirc' else oline[v]['text'].strip().strip('.')
            psent = pline[v].strip().strip(
                '.') if task != 'multirc' else pline[v]['text'].strip().strip('.')
            # osent = re.split(r'[;.?]', osent)
            osent = re.split(r'[.]', osent)
            psent = re.split(r'[.]', psent)
            for oss, pss in zip(osent, psent):
                pss = re.sub(r' +', ' ', pss)
                oss = re.sub(r' +', ' ', oss)
                item_str = oss + '|' + pss + '|' + str(item) + '|' + v
                item_in_csv.append(item_str)
        item += 1

    return item_in_csv


def argparser():

    import argparse

    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--shuffle', default=0, required=False,
                        help='create a permuted version of datasets')
    parser.add_argument('--score', default=1, required=False,
                        help='calculate distance between permuted and original sents')
    parser.add_argument('--csv', default=1, required=False,
                        help='create csv for rnds')

    arguments = parser.parse_args()

    return arguments
