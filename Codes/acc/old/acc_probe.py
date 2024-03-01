# calculate the probe acc
# superGlue + Glue
# Three main funcs
# 1 permute sents

import os
import logging
import pandas as pd
from subprocess import check_call
from textdistance import levenshtein as ld


def define_logger():
    '''
    logger config
    '''
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def score(sent1, sent2):
    ''' score two strings with levenshtein distance
    '''
    norm_score = ld.normalized_similarity(sent1, sent2)
    return norm_score


def read_perturbation(fpath):
    ''' pass in a csv file 
    save and force write on the original file
    '''
    df = pd.read_csv(fpath, header=0, index_col=0, dtype={
                     "Generated Text": "string", "Actual Text": "string"})
    df = df.dropna()  # this is weird

    df['probe-acc'] = df.apply(lambda x: score(x['Generated Text'],
                               x['Actual Text']), axis=1)

    df.to_csv(fpath)


def get_mean_score(fpath):
    '''
    get mean score for each sent
    '''
    df = pd.read_csv(fpath, header=0, index_col=0, dtype={
                     "Generated Text": "string", "Actual Text": "string", "item_num": int})
    df = df.dropna()  # this is weird
    # df['item_num'] = df['item_num'].astype('str')
    item_score = df.groupby('item_num')['probe-acc'].mean()

    return item_score


def loop_and_mean(path):
    '''
    loop over the three random rounds, and get the mean probe acc for each item (not sent) for each dataset
    '''

    tasks = ['boolq', 'cb', 'copa', 'mrpc', 'multirc',
             'qqp', 'rte', 'wic', 'winogrande', 'wnli', 'wsc']
    dataset_scores = []
    for task in tasks:
        # create dir
        round_scores = []
        # for rx in ['r1', 'r2', 'r3']:
        # name = rx+'-'+task+'-'+'pmt'+'.csv'
        name = task+'-'+'pmt'+'.csv'
        fpath = os.path.join(path, name)
        item_score = get_mean_score(fpath)
        round_scores = item_score
        # if rx == 'r1':
        #     round_scores = item_score
        # else:
        #     round_scores += item_score
        # round_scores = round_scores / 3 # 3 loops here
        round_scores = round_scores.to_frame()
        round_scores.to_csv(os.path.join(path, task+'.csv')
                            )  # save to a csv file

    return dict(zip(tasks, dataset_scores))


# def add_probe_acc_json():
#     '''
#     read each json and add probe acc to the original file
#     '''


def main():
    # load logging formatter
    define_logger()

    path = '/Users/xdchen/Downloads/data/results'
    # csv file format
    # rx-task-pmt
    # for rx in ['r1', 'r2', 'r3']:
    for task in ['boolq', 'cb', 'copa', 'mrpc', 'multirc', 'qqp', 'rte', 'wic', 'winogrande', 'wnli', 'wsc']:
        # name = rx+'-'+task+'-'+'pmt'+'.csv'
        name = task+'-'+'pmt'+'.csv'
        fpath = os.path.join(path, name)
        logging.info(fpath)
        read_perturbation(fpath)

    loop_and_mean(path)


if __name__ == '__main__':
    main()
