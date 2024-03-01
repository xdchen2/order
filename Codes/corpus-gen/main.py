# permute_json sentences with 1-gram
# superGlue
# Three main funcs
# 1 permute_json sents
# 2 compute distance between permutation and original sents (control acc)
# 3 save the computed score to the orginal json file from which the original sents were extracted

import os
import re
import pandas as pd
import random
import json
from scramble import control_acc, permute_json, create_pmt, bimute_json
from textdistance import levenshtein as ld
from para import params


def shuffle_json(inpath, otpath, task):
    ''''
    FUN1
    '''
    if task != 'winogrande':
        files = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    else:
        files = ['train_xl.jsonl', 'dev.jsonl', 'test.jsonl']

    for splits in files:
        fpth = os.path.join(inpath, task, splits) if task != 'winogrande' else os.path.join(
            inpath, task, 'winogrande_1.1', splits)
        opth = os.path.join(otpath, task, splits) if task != 'winogrande' else os.path.join(
            otpath, task, 'winogrande_1.1', splits)
        task_content_list = permute_json(fpth, task)
        with open(opth, 'w+') as foutput:
            for line in task_content_list:
                json.dump(line, foutput)
                foutput.write('\n')

    return None


def biinput(inpath, otpath, task):
    '''
    RND json + ORG json
    '''
    if task != 'winogrande':
        files = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    else:
        files = ['train_xl.jsonl', 'dev.jsonl', 'test.jsonl']

    for splits in files:
        fpth = os.path.join(inpath, task, splits) if task != 'winogrande' else os.path.join(
            inpath, task, 'winogrande_1.1', splits)
        opth = os.path.join(otpath, task, splits) if task != 'winogrande' else os.path.join(
            otpath, task, 'winogrande_1.1', splits)
        task_content_list = bimute_json(fpth, task)
        with open(opth, 'w+') as foutput:
            for line in task_content_list:
                json.dump(line, foutput)
                foutput.write('\n')



def eval_control(inpath, otpath, task):
    ''''
    FUN3
    '''

    source_json = os.path.join(inpath, task, 'val.jsonl') if task != 'winogrande' else os.path.join(
        inpath, task, 'winogrande_1.1', 'dev.jsonl')
    target_json = os.path.join(otpath, task, 'val.jsonl') if task != 'winogrande' else os.path.join(
        otpath, task, 'winogrande_1.1', 'dev.jsonl')
    # stdout_json = os.path.join(otpath, task, 'score_val.jsonl') if task != 'winogrande' else os.path.join(
        # otpath, task, 'winogrande_1.1', 'score_val.jsonl')
    stdout_json = target_json

    item_injson = control_acc(target_json, source_json, task)

    with open(stdout_json, 'w+') as foutput:
        for line in item_injson:
            json.dump(line, foutput)
            foutput.write('\n')

    return None


def extract_pmts(inpath, otpath, task):
    ''''
    FUN2
    '''

    source_json = os.path.join(inpath, task, 'val.jsonl') if task != 'winogrande' else os.path.join(
        inpath, task, 'winogrande_1.1', 'dev.jsonl')
    target_json = os.path.join(otpath, task, 'val.jsonl') if task != 'winogrande' else os.path.join(
        otpath, task, 'winogrande_1.1', 'dev.jsonl')
    stdout_json = os.path.join(otpath, task, task+'-'+'pmt.csv') if task != 'winogrande' else os.path.join(
        otpath, task, 'winogrande_1.1', task+'-'+'pmt.csv')

    item_injson = create_pmt(target_json, source_json, task)
    with open(stdout_json, 'w+') as foutput:
        for line in item_injson:
            foutput.write(line)
            foutput.write('\n')

    return None


def otpath_eval():

    '''
    Test whether the file exists
    '''
    # if os.path.exists(otpath):
    #     pass
    # else:
    #     os.mkdir(otpath)
    os.makedirs(otpath, mode = 0o777, exist_ok = True) 

    # create nested file lists
    for task in params['TASK'].keys():
        if task != 'winogrande':
            if not os.path.exists(os.path.join(otpath, task)):
                os.mkdir(os.path.join(otpath, task))
        else:
            if not os.path.exists(os.path.join(otpath, task)):
                os.mkdir(os.path.join(otpath, task))
            if not os.path.exists(os.path.join(otpath, task, 'winogrande_1.1')):
                os.mkdir(os.path.join(otpath, task, 'winogrande_1.1'))

    return None




if __name__ == '__main__':

    random.seed(params['SEED'])

    # orginal file path
    inpath = params['DATA_PATH']
    # perumute file path
    otpath = params['OUTS_PATH']

    otpath_eval()

    # task datasets/files in the same format of the source dataset/file
    for task in params['TASK'].keys():
        
        if params['M-CREATE']:
            shuffle_json(inpath, otpath, task)
        if params['M-BiInput']:
            biinput(inpath, otpath, task)
        if params['M-EVAL']:
            eval_control(inpath, otpath, task)
        if params['M-PMTS']:
            extract_pmts(inpath, otpath, task)
