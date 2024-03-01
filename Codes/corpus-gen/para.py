
# define model parameters

params = {
    "M-CREATE": 0, # create permuted jsons
    "M-BiInput": 1, # create bi-input rnd + org
    "M-EVAL": 0, # control score and append to json
    "M-PMTS": 0, # create pmt from json
    'SEED':6, # seed num
    "DATA_PATH": '/Users/xdchen/Downloads/eval2/data/data-agg', # has to be a dir
    "OUTS_PATH": '/Users/xdchen/Downloads/bin/rnd6/data', # has to be a dir
    "TASK": {
         'boolq': ['passage', 'question'],
         'cb': ['premise', 'hypothesis'], 
         'copa': ['choice1', 'choice2', 'premise'], 
         'mrpc': ['text_a', 'text_b'], 
         'multirc': ['passage'], 
        #  'wic': ['sentence1', 'sentence2'], 
         'winogrande': ['sentence'], 
        #  'wnli': ['premise', 'hypothesis'], 
        #  'wsc': ['text'], 
         'qqp': ['text_a', 'text_b'],
         'rte': ['premise', 'hypothesis'],
         'sst': ['text'],
         }
}