# convert json to csv 

import json
import string
import os
import re
import random
import pandas as pd
from random import shuffle

def permute(sent):
    ''' input a single sent list
    out: list of permuted sent
    '''
    shuffle(sent)

    return sent

def json_read(file_path):
    '''
    transform json to dic
    '''
    with open(file_path, 'r') as f:
        # data = f.readlines()
        data = [json.loads(line) for line in f]

    # data = [json.load(line.strip()) for line in data]

    return data



def remove_punc(sent):
    '''
    take one sent and remove all puncts
    '''
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    return sent


def wino(json, permute_num=5):
    '''
    specific
    '''
    data = json_read(json)

    sent1 = [re.sub('_', i['option1'], i['sentence']) for i in data]
    sent2 = [re.sub('_', i['option2'], i['sentence']) for i in data]

    sents = sent1 + sent2

    sents = list(set(sents))

    sents = [remove_punc(sent) for sent in sents]

    permu_data = []
    
    # permute 5 times
    for i in range(permute_num):

        # change seed once entering a new loop
        random.seed(5*i)

        permuted_sentences = [' '.join(permute(line.split(' '))) for line in sents]

        permu_data.append(permuted_sentences)

    permute_dataframe = pd.DataFrame({"source": sents, 
                        "target1": permu_data[0], 
                        "target2": permu_data[1], 
                        "target3": permu_data[2], 
                        "target4": permu_data[3], 
                        "target5": permu_data[4]}
                        )

    return permute_dataframe


def extract_from_json(json, key, permute_num=5):
    '''
    read in a json file
    output dataframe with permuted sentences
    args:
        json - the path to the json file
        key  - key words to index 
    '''

    data = json_read(json)

    clean_data = []

    for line in data:

        # select the text line
        line = line[key].strip()
        # split each sent
        line = line.split('.')
        # split each sent
        line = [sl for sl in line if sl != '']
        # remove all punctuations
        line = [remove_punc(sl) for sl in line]

        # append to a new list
        clean_data.append(line)

    clean_data = [line for seq in clean_data for line in seq]
    permu_data = []
    
    # permute 5 times
    for i in range(permute_num):

        # change seed once entering a new loop
        random.seed(5*i)

        permuted_sentences = [' '.join(permute(line.split(' '))) for line in clean_data]

        permu_data.append(permuted_sentences)

    permute_dataframe = pd.DataFrame({"source": clean_data, 
                        "target1": permu_data[0], 
                        "target2": permu_data[1], 
                        "target3": permu_data[2], 
                        "target4": permu_data[3], 
                        "target5": permu_data[4]}
                        )

    return permute_dataframe





def main():
    '''
    '''

    #         WiC 'sentence1', 'sentence2' 
    #         WSC 'text', 
    #         ReCoRD 'passage'['text'], # longer texts
    #         RTE 'premise', 'hypothesis', 
    #         MultiRC 'passage'['text'], # longer texts
    #         COPA 'premise', 'choice1', 'choice2', 
    #         CB 'premise', 'hypothesis', 
    #         BoolQ 'passage', 'question', # longer texts
    #         AX-g 'hypothesis', 'premise', 
    #         AX-b 'sentence1', 'sentence2',
    #         Winogrande 'sentence'

    # hyperparameters
    job = 'Winogrande'
    key = 'sentence'
    outname = 'Winogrande.csv'
    file_path = '/Users/xdchen/Downloads/superGlue'

    in_path = os.path.join(file_path, job, 'test.jsonl')
    ot_path = os.path.join(file_path, job, outname)

    df = wino(in_path)

    # df = extract_from_json(in_path, key)

    print(df.head())

    df.to_csv(ot_path, index=0)



if __name__ == '__main__':

    # hyperparameters

    main()