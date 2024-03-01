import torch
import pandas as pd
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from loader import NLIClass, SDPClass
from torch.utils.data import DataLoader


def T5Test(df, model_params, source_text="shuf_sent", target_text="sent", device='cpu', output_file='./test.csv'):

    """
    Test the probe model on samples
    Format -- 
    Generated Text; Actual Text etc.
    """

    # df['source'] = df.iloc[:,0] + " </s>"
    # df['target'] = "recover from perturbation: " + df.iloc[:,1] + " </s>"

    df['source'] = df['org'] + " </s>"
    df['target'] = "recover from perturbation: " + df['rnd'] + " </s>"
    
    # The task is shuffled (target) -> Restored (source)
    source_text = "target"
    target_text = "source"

    val_dataset = df[["target", "source"]]

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    val_set = SDPClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    val_loader = DataLoader(val_set, **val_params)

    preds = []
    tagts = []

    for epoch in range(model_params["VAL_EPOCHS"]):
        pred, tagt = validate(epoch, tokenizer, model, device, val_loader)
        preds += pred
        tagts += tagt

    pred_csv = pd.DataFrame({"gnt": preds, "org": tagts})
    pred_csv['rnd'] = df['rnd']
    pred_csv['inum'] = df['inum']
    pred_csv['key'] = df['keys']

    pred_csv.to_csv(output_file, sep='|', index=False)

    return pred_csv
    

def validate(epoch, tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            if _%10==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
            
    return predictions, actuals