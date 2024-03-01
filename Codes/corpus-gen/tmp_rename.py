from subprocess import check_call
import os

# check_call([args.praat, './praat/tune.praat', single_grid.path])

tasd = 'r5'
path = '/Users/xdchen/Downloads'

path = os.path.join(path, tasd)

tasks = {'boolq': ['passage', 'question'],
         'cb': ['premise', 'hypothesis'], 
         'copa': ['choice1', 'choice2', 'premise'], 
         'mrpc': ['text_a', 'text_b'], 
         'multirc': ['passage'], 
         'wic': ['sentence1', 'sentence2'], 
         'winogrande': ['sentence'], 
         # winogrande: sentence '_' option1 option2
         'wnli': ['premise', 'hypothesis'], 
         'wsc': ['text'], 
         'qqp': ['text_a', 'text_b'],
         }

for k in tasks.keys():
    inpath = os.path.join(path, k) if k != 'winogrande' else os.path.join(path, k, 'winogrande_1.1')
    filpath = os.path.join(path, k, 'pmt.csv') if k != 'winogrande' else os.path.join(path, k, 'winogrande_1.1', 'pmt.csv')
    oupath = os.path.join(path, tasd+'-'+k+'-'+'pmt.csv')
    check_call(['mv', filpath, oupath])
    check_call(['rm', '-rf', inpath])