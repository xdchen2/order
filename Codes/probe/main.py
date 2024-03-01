import pandas as pd
import sys
import os
from para import model_params
from train import T5Trainer
from eval import T5Test

def train():
    '''
    Train a probe model
    '''
    path = PARAS['FPATH']
    df = pd.read_csv(path, sep='|', header=0, warn_bad_lines=True, error_bad_lines=False, nrows=100000)

    T5Trainer(
        dataframe=df,
        source_text="shuf_sent",
        target_text="sent",
        model_params=model_params,
        output_dir='./',
        output_file=model_params["DATA_PATH"],
        device=device
        )

def eval():
    '''
    Eval datasets
    '''

    for tname in PARAS['TASKS']:

        path = os.path.join(PARAS['FPATH'], 'data', tname, '{}-pmt.csv'.format(tname))
        df = pd.read_csv(path, sep='|', header=0, warn_bad_lines=True, error_bad_lines=False, nrows=100000)

        T5Test(
            df, 
            model_params,
            source_text="shuf_sent",
            target_text="sent",
            output_file=path,
            device=device
            )


if __name__ == '__main__':
    # Setting up the device for GPU usage
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    PARAS = {
        'FPATH': sys.argv[1], # path to probe files
        'MPATH': './model', # path to model files
        'OPATH': './out', # path to outputs
        'TASKS':  ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'rte' , 'wic' , 'winogrande' , 'qqp' , 'wsc'],
    }

    # output file name
    csv_name = 'pred.csv' 

    if model_params["MODE"] == 'train':
        train()

    if model_params["MODE"] == 'eval':
        eval()
