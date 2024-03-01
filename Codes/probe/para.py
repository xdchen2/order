# define model parameters specific to conditional T5
model_params = {
    "MODE": "eval",
    "DATA_PATH": "/Users/xdchen/Downloads/rte-pmt.csv",
    "MODEL": "/home/mila/x/xuanda.chen/probe/model",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 32,  # training batch size
    "VALID_BATCH_SIZE": 32,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}