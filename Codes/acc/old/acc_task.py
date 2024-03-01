import os, re
import pandas as pd
import torch
import json


def read_pred(fpath):
    '''read in pred file
    return a dataframe
    '''
    preds = torch.load(fpath)
    task_list = []
    idxs_list = []
    pred_list = []
    for task, pred in preds.items():
        for guid, pred in zip(list(pred["guids"]), list(pred["preds"])):
            # format: boolq test-3237 1
            task_list.append(task)
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            pred_list.append(pred)
            # print(task, guid.split('-')[1], pred, sep="\t")
    # df = pd.DataFrame({'dataset':task_list, 'idx':idxs_list, 'pred': pred_list})

    return task_list, idxs_list, pred_list


def get_rounds(dfall, dfsub, rx):
    '''
    put three rounds in one file'''

    if len(dfall) == 0:
        dfall = dfsub
    else:
        tag = 'pred-'+rx
        dfall[tag] = dfsub['pred']

    return dfall


def get_popular_ans(anslist):
    ''' pass in rounds
    vote for an optimal answer
    '''
    return max(anslist, key=anslist.count)


def read_json(path):
    ''' read in data
    '''
    with open(path, 'r') as f: 
        data = [json.loads(line) for line in f]
    
    return data


def get_pred_lab(path):
    '''from the pred file
    '''
    tasks = ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'qqp' , 'rte' , 'wic' , 'winogrande' , 'wnli' , 'wsc']

    pred_dict = {}

    for task in tasks:
        vfpath = os.path.join(path, task, 'val_preds.p')
        dataset, idx, pred = read_pred(vfpath)

        pred_dict[task] = [pred, idx, dataset]

    return pred_dict



def get_true_lab(path):
    '''get true labels for each task from json files
    '''
    tasks = ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'qqp' , 'rte' , 'wic' , 'winogrande' , 'wnli' , 'wsc']
    labns = ['label', 'label', 'label', 'label', 'passage', 'label', 'label', 'label', 'answer', 'label', 'label']
    labskeys = dict(zip(tasks, labns))

    trans_lab = {
                'boolq': {True: 1, False: 0},
                'cb': {'neutral': 1, 'contradiction': 1, 'entailment': 0},
                'copa': {0:0, 1:1},
                'mrpc': {'0':0, '1':1},
                'multirc': {0:0, 1:1},
                'qqp': {'0':0, '1':1},
                'rte': {'not_entailment': 1, 'entailment': 0},
                'wic': {True: 1, False: 0},
                'winogrande': {'1':0, '2':1},
                'wnli': {'0':0, '1':1},
                'wsc': {True: 1, False: 0},
                }

    n = 'val.jsonl'

    lab_dict = {}

    for t in tasks:
        dspath = os.path.join(path, t, n) if t != 'winogrande' else os.path.join(path, t, 'winogrande_1.1', n)
        jdata = read_json(dspath)

        if t != 'multirc':
            labs = [item[labskeys[t]] for item in jdata] 
        else:
            # takes a while to figure out where to locate the labs in multirc
            labs = [rx['label'] for item in jdata for text in item['passage']['questions'] for rx in text['answers']]

        labs = [trans_lab[t][i] for i in labs]

        # print(t, labs[0:5])

        lab_dict[t] = labs

    return lab_dict


def save_pred_true(pred, true):
    '''
    save as dataframe in a csv
    '''

    df = pd.DataFrame()
    tasks = ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'qqp' , 'rte' , 'wic' , 'winogrande' , 'wnli' , 'wsc']

    for t in tasks:
        # assert length
        if len(pred[t][0]) != len(true[t]):
            print(t, 'error: number not aligned')

        tmpdf = pd.DataFrame({'dataset':pred[t][2], 'idx':pred[t][1], 'pred': pred[t][0], 'true': true[t]})

        if  len(df) == 0:
            df = tmpdf
        else:
            df = pd.concat([df, tmpdf])

    df.to_csv('/Users/xdchen/Downloads/data/permute_val_pred_true.csv')

        


def get_pred():

    # org_path = '/Users/xdchen/Downloads/data/org'
    # pmt_path = '/Users/xdchen/Downloads/data/pmt'

    dat_path = '/Users/xdchen/Downloads/data/runs-seed'

    final_df = pd.DataFrame()

    tasks = ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'qqp' , 'rte' , 'wic' , 'winogrande' , 'wnli' , 'wsc']

    for task in tasks:
        # itemodf = pd.DataFrame()
        # itempdf = pd.DataFrame()
        itemvdf = pd.DataFrame()
        # for rx in ['v1', 'v2', 'v3']:
        for rx in ['v1']:
            # ofpath = os.path.join(org_path, 'run-'+rx, task, 'test_preds.p')
            # pfpath = os.path.join(pmt_path, 'pmt-'+rx, task, 'test_preds.p')
            vfpath = os.path.join(dat_path, task, 'val_preds.p')
            vdf = read_pred(vfpath)
            itemvdf = get_rounds(itemvdf, vdf, rx)

            # odf = read_pred(ofpath)
            # pdf = read_pred(pfpath)

            # itemodf = get_rounds(itemodf, odf, rx)
            # itempdf = get_rounds(itempdf, pdf, rx)

        # itemvdf['fpred'] = itempdf.apply(lambda x: get_popular_ans([x['pred'], x['pred-v2'], x['pred-v3']]), axis=1)

        if  len(final_df) == 0:
            final_df = itemvdf
        else:
            final_df = pd.concat([final_df, itemvdf])

    final_df.to_csv('/Users/xdchen/Downloads/data/permute_val_pred_true.csv')


if __name__ == '__main__':
    pred_path = '/Users/xdchen/Downloads/data/eval-run-permuted'
    true_path = '/Users/xdchen/Downloads/data/eval2data'

    pred = get_pred_lab(pred_path)
    true = get_true_lab(true_path)
    save_pred_true(pred, true)
    print('file saved')


