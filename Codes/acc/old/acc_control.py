import os
import pandas as pd
import json

from textdistance import levenshtein as ld


def score(sent1, sent2):
    ''' score two strings with levenshtein distance
    '''
    norm_score = ld.normalized_similarity(sent1, sent2)
    return norm_score

def acc_compute(path):
    '''
    Compute based on '''

    df = pd.read_csv(path, header=0, sep=',;')
    df['idx'] = df.index
    df['sentlen'] = df['orginal sent'].apply(lambda x: len(x.strip().split(' ')))
    # df['control']

    return None





def json_read(jpath):

    with open(jpath, 'r') as f: 
        jdata = [json.loads(line) for line in f]

    return jdata


def mean_control_acc(path):
    '''
    read in a json file and get the mean value
    '''
    tasks = ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'qqp' , 'rte' , 'wic' , 'winogrande' , 'wnli' , 'wsc']

    for task in tasks:
        task_score = []
        # for rx in ['r1', 'r2', 'r3']:
        # round_dir = os.path.join(path, 'data-'+rx)
        round_dir = os.path.join(path, 'data-r1')
        if task != 'winogrande':
            datst_dir = os.path.join(round_dir, task, 'score_val.jsonl')
        else:
            datst_dir = os.path.join(round_dir, task, 'winogrande_1.1', 'score_val.jsonl')
        jdata = json_read(datst_dir)
        jsidx = [i for i, item in enumerate(jdata)]
        jscor = [item['score'] for i, item in enumerate(jdata)]
        task_score.append(jscor)
        
        # final = []
        # for a, b, c in zip(task_score[0], task_score[1], task_score[2]):
        #     final.append((a+b+c)/3)

        df = pd.DataFrame({'control-acc': jscor, 'idx': jsidx})
        opath = os.path.join(round_dir, task+'.csv')
        df.to_csv(opath)


def main():
    path = '/Users/xdchen/Downloads/data/perturb-data-json/'
    mean_control_acc(path)


if __name__ == '__main__':
    main()

