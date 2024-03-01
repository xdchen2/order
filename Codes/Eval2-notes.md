# CC or MILA

## interactive compute node
salloc --gres=gpu:1 -c 4 --mem=24000
## Load virtual environ
module load python/3.8
source ENV/bin/activate
cd baby/
cd jiant/

## Load conda
<!-- ? conda activate "simple" -->
module load anaconda/3
conda activate bb

## Eval pipeline
module load anaconda/3
conda activate eval

cd ../evaluation-pipeline/; conda activate eval; python babylm_eval.py '/home/mila/x/xuanda.chen/lm/vc-pcfg/vpcfglm/val' 'decoder'

cd ../vpcfglm/; conda activate bb; python train.py

## zero shot
python babylm_eval.py '/home/mila/x/xuanda.chen/lm/vc-pcfg/vpcfglm/val' 'decoder'
python babylm_eval.py '/network/scratch/x/xuanda.chen/baby_model' 'decoder'
python babylm_eval.py './models' 'decoder' -t "blimp"
python babylm_eval.py './models' 'encoder' -t "blimp"
## fine-tuning
./finetune_all_tasks.sh '/network/scratch/x/xuanda.chen/baby_model'
./finetune_all_tasks.sh './models'
./finetune_all_tasks.sh '../vpcfg/models'


from minicons import scorer
ilm_model = scorer.IncrementalLMScorer('models', 'cpu')
stimuli = ["The keys to the cabinet are on the table.", "The keys to the cabinet is on the table."]

print(ilm_model.sequence_score(stimuli, reduction = lambda x: x.mean(0).item()))

print(ilm_model.sequence_score(stimuli, reduction = lambda x: x.sum(0).item()))



../vpcfg/models/

## Git clone
<!-- git remote add origin git@github.com:evaportelance/babylm-joint-learning.git -->
git add .
git commit -m ''
git push -u origin main

# Train tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets

with open('/network/scratch/x/xuanda.chen/babylm_data/train.txt' ) as f:
    lines = f.readlines()

dataset = {'text': [l.strip() for l in lines]}
dataset = datasets.Dataset.from_dict(dataset)

def batch_iterator():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

tokenizer = AutoTokenizer.from_pretrained("babylm/opt-125m-strict")
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=8192)
new_tokenizer.save_pretrained('../vpcfg/models')





python train.py --tiny --data_path "../preprocessed-data/abstractscenes" --prefix 'all' --num_epochs 3
python train.py --tiny --data_path "../preprocessed-data/abstractscenes" --prefix 'all' --num_epochs 3 --visual_mode VISUAL_MODE


print(mlm_model.sequence_score(stimuli, reduction = lambda x: x.mean(0).item()))
print(mlm_model.sequence_score(stimuli, reduction = lambda x: x.sum(0).item()))


# JIANT

## data download

python jiant/scripts/download_data/runscript.py \
    download \
    --tasks sst \
    --output_path /network/scratch/x/xuanda.chen/tasks/

## jiant training
<!-- https://github.com/nyu-mll/jiant/blob/master/guides/tasks/supported_tasks.md -->
<!-- boolq  cb  copa  mrpc  multirc  qqp  record(needs to be 3 epochs)  rte  wic  winogrande  wnli  wsc -->

python jiant/proj/simple/runscript.py \
    run \
    --run_name cb \
    --tasks cb \
    --exp_dir /network/scratch/x/xuanda.chen/tasks/ \
    --data_dir /network/scratch/x/xuanda.chen/tasks \
    --hf_pretrained_model_name_or_path roberta-base \
    --learning_rate 1e-5 \
    --train_batch_size 16 \
    --num_train_epochs 10 \
    --write_val_preds \
    --do_save_best \
    --seed 42
<!-- run_std works fine the data dir matters -->

for TASK in boolq cb copa mrpc multirc qqp rte wic winogrande wnli wsc;
do
python jiant/proj/simple/runscript.py \
    run \
    --run_name $TASK \
    --exp_dir /network/scratch/x/xuanda.chen/tasks/pmt3 \
    --hf_pretrained_model_name_or_path roberta-base \
    --data_dir /network/scratch/x/xuanda.chen/tasks/pmt3 \
    --tasks $TASK \
    --learning_rate 1e-5 \
    --train_batch_size 16 \
    --num_train_epochs 10 \
    --write_test_preds \
    --seed 45;
done;

    <!-- # do not want models saved -->
    <!-- --do_save_best \ -->

## formatter
python jiant/scripts/benchmarks/benchmark_submission_formatter.py \
    --benchmark SUPERGLUE \
    --tasks rte boolq cb copa multirc wsc wic \
    --input_base_path /network/scratch/x/xuanda.chen/runs \
    --output_path /network/scratch/x/xuanda.chen/runs

python jiant/scripts/benchmarks/benchmark_submission_formatter.py \
    --benchmark GLUE \
    --tasks mrpc qqp wnli \
    --input_base_path /network/scratch/x/xuanda.chen/runs \
    --output_path /network/scratch/x/xuanda.chen/runs

### formatter for items in preds (wino)
import torch
all_preds = torch.load('')
for task, pred_dict in all_preds.items():
    for guid, pred in zip(list(pred_dict["guids"]), list(pred_dict["preds"])):
        print(task, guid, pred, sep="\t")

## collect eval data and transfer
find ./ -name '*val_metrics.json' | xargs -i{} scp -r mila:{} /Users/xdchen/Downloads/

## transfer prediction from temp server to local
scp -r mila:/home/mila/x/xuanda.chen/scratch/runs /Users/xdchen/Downloads/
scp -r /Users/xdchen/Downloads/rnd.zip mila:/network/scratch/x/xuanda.chen/

find /home/mila/x/xuanda.chen/scratch/bin -name '*json' -exec sed -i 's/rnd\/rnd/bin\/rnd/g' {} +
find /home/mila/x/xuanda.chen/scratch/rnd -name '*.csv' -exec sed -i 's/"p(s|t)"/cp/g' {} +
df = df.round(4)

for i in `find ./ -name 'val_preds.p'`; do RND=${i:2:4}; FNAME=$(basename ${i}); FPATH=$(dirname ${i}); DNAME=$(basename ${FPATH}); echo ${RND}-${DNAME}-${FNAME}; done


## check what has been done
ls scratch/runs/

## check ongoing work
squeue -u xuanda.chen

# Data

## data format
mrpc: text_a text_b
qqp: text_a text_b
wnli: premise hypothesis
winogrande: sentence '_' option1 option2
multirc: passage['text']
wic: sentence1 sentence2
cb: premise hypothesis
rte: premise hypothesis
copa: choice1 choice2 premise
wsc: text
boolq: passage question


## usage

usage: runscript.py [-h] [--ZZsrc ZZSRC] [--ZZoverrides ZZOVERRIDES [ZZOVERRIDES ...]]
                    --run_name RUN_NAME --exp_dir EXP_DIR --data_dir DATA_DIR
                    --hf_pretrained_model_name_or_path HF_PRETRAINED_MODEL_NAME_OR_PATH
                    [--model_weights_path MODEL_WEIGHTS_PATH]
                    [--model_cache_path MODEL_CACHE_PATH] [--tasks TASKS]
                    [--train_tasks TRAIN_TASKS] [--val_tasks VAL_TASKS]
                    [--test_tasks TEST_TASKS] [--train_batch_size TRAIN_BATCH_SIZE]
                    [--max_seq_length MAX_SEQ_LENGTH] [--num_train_epochs NUM_TRAIN_EPOCHS]
                    [--train_examples_cap TRAIN_EXAMPLES_CAP] [--create_config] [--do_save]
                    [--do_save_last] [--do_save_best] [--write_val_preds]
                    [--write_test_preds] [--eval_every_steps EVAL_EVERY_STEPS]
                    [--save_every_steps SAVE_EVERY_STEPS]
                    [--save_checkpoint_every_steps SAVE_CHECKPOINT_EVERY_STEPS]
                    [--no_improvements_for_n_evals NO_IMPROVEMENTS_FOR_N_EVALS]
                    [--keep_checkpoint_when_done] [--force_overwrite] [--seed SEED]
                    [--learning_rate LEARNING_RATE] [--adam_epsilon ADAM_EPSILON]
                    [--max_grad_norm MAX_GRAD_NORM] [--optimizer_type OPTIMIZER_TYPE]
                    [--no_cuda] [--fp16] [--fp16_opt_level FP16_OPT_LEVEL]
                    [--local_rank LOCAL_RANK] [--server_ip SERVER_IP]
                    [--server_port SERVER_PORT]


rm ./pmt3/configs/*; cp -r ./pmt1/configs/* ./pmt3/configs/

find ./pmt2 -name '*config.json*' -exec sed -i 's/pmt1/pmt2/' {} +

## wino problem
cp ~/scratch/tasks/data/winogrande/winogrande_1.1/train_xl-labels.lst ~/scratch/tasks/pmt1/data/winogrande/winogrande_1.1/train_xl-labels.lst
cp ~/scratch/tasks/data/winogrande/winogrande_1.1/dev-labels.lst ~/scratch/tasks/pmt1/data/winogrande/winogrande_1.1/dev-labels.lst

## wic problem

## rte problem
scp -r /Users/xdchen/Downloads/data-r5/rte mila:/home/mila/x/xuanda.chen/scratch/tasks/pmt5/data

## wsc problem
cp -r /network/scratch/x/xuanda.chen/cache/roberta/wic/val/* /network/scratch/x/xuanda.chen/tasks/pmt1/cache/roberta/wic/val/;
mkdir  /network/scratch/x/xuanda.chen/tasks/pmt1/cache/roberta/wsc/val_labels;
cp -r /network/scratch/x/xuanda.chen/cache/roberta/wsc/val_labels/* /network/scratch/x/xuanda.chen/tasks/pmt1/cache/roberta/wsc/val_labels/

cp -r /network/scratch/x/xuanda.chen/cache/roberta/wsc/val/* /network/scratch/x/xuanda.chen/tasks/pmt2/cache/roberta/wsc/val/;
mkdir  /network/scratch/x/xuanda.chen/tasks/pmt2/cache/roberta/wsc/val_labels;
cp -r /network/scratch/x/xuanda.chen/cache/roberta/wsc/val_labels/* /network/scratch/x/xuanda.chen/tasks/pmt2/cache/roberta/wsc/val_labels/

cp -r /network/scratch/x/xuanda.chen/cache/roberta/wsc/val/* /network/scratch/x/xuanda.chen/tasks/pmt3/cache/roberta/wsc/val/;
mkdir  /network/scratch/x/xuanda.chen/tasks/pmt3/cache/roberta/wsc/val_labels;
cp -r /network/scratch/x/xuanda.chen/cache/roberta/wsc/val_labels/* /network/scratch/x/xuanda.chen/tasks/pmt3/cache/roberta/wsc/val_labels/


## config

{
  "task": "cb",
  "paths": {
    "train": "/network/scratch/x/xuanda.chen/tasks/pmt1/data/cb/train.jsonl",
    "test": "/network/scratch/x/xuanda.chen/tasks/pmt1/data/cb/test.jsonl",
    "val": "/network/scratch/x/xuanda.chen/tasks/pmt1/data/cb/val.jsonl"
  },
  "name": "cb"
}

## MSC

for i in `find ./ -name 'val_preds.p'`; do RND=${i:2:4}; FNAME=$(basename ${i}); FPATH=$(dirname ${i}); DNAME=$(basename ${FPATH}); echo ${RND}-${DNAME}-${FNAME}; done

cp $i ../output/${RND}-${DNAME}-${FNAME}

python runproc.py winogrande bin/rnd6

find /network/scratch/x/xuanda.chen/bin/rnd6/data/winogrande -name 'train_xl.jsonl' | xargs -i{} sed -i 's/_//' {}


## pred check

14|1|0|cb|1|0|0|0|0|0
{"premise": "And I don't want to have to lie to them. The kidnappers have given us until October the eleventh to deliver the document and I haven't despaired of finding it before then. But if the police learn I 've been to America they 'll ask why.", "hypothesis": "he's been to America", "label": "entailment", "idx": 13}
RND2
{"premise": "don't I have want to lie And them to to. and document us given of eleventh The haven't to it I the then until  before have October finding deliver the despaired kidnappers. the been  I to police America learn 've if why they 'll ask But. ", "hypothesis": "America been he's to", "label": "entailment", "idx": 13, "score": 0.33999076692249985}
RND1
{"premise": "have I lie to want them to And to don't. haven't October us kidnappers the  then despaired I The the have finding to it eleventh of deliver and before document given until. But if they  police ask 'll I learn why 've been America to the. ", "hypothesis": "been he's to America", "label": "entailment", "idx": 13, "score": 0.4750221209148441}
2|0|1|cb|1|0|0|0|1|1
{"premise": "``Who knows? The point is, do we go with it or not?'' Do we assume there is a shipment?", "hypothesis": "there is a shipment", "label": "neutral", "idx": 1}
2|0|1|multirc|1|1|1|1|1|1
933|0|0|boolq|1|1|1|1|1|1
{"idx": 932, "label": false, "passage": "Cremaster muscle -- The cremaster muscle is a muscle that covers the testis and the spermatic cord.", "question": "is the cremaster muscle in the spermatic cord"}
RND2
{"idx": 932, "label": false, "passage": "is muscle the covers and that -- The spermatic the cremaster Cremaster muscle muscle a testis cord. ", "question": "the muscle cord cremaster is the spermatic in", "score": 0.3812925170068027}

917|1|1|boolq|1|1|0|0|1|1
{"idx": 916, "label": true, "passage": "Stand-your-ground law -- The states that have legislatively adopted stand-your-ground laws are Alabama, Alaska, Arizona, Florida, Georgia, Idaho, Indiana, Iowa, Kansas, Kentucky, Louisiana, Michigan, Mississippi, Missouri, Montana, Nevada, New Hampshire, North Carolina, Oklahoma, Pennsylvania, South Carolina, South Dakota, Tennessee, Texas, Utah, West Virginia and Wyoming.", "question": "is there a self defense law in kentucky"}
1037|1|0|boolq|0|0|1|1|0|0
{"idx": 1036, "label": false, "passage": "Jaws (ride) -- On December 2, 2011, Universal Orlando Resort announced that the Jaws attraction along with the entire Amity area of Universal Studios Florida would close permanently on January 2, 2012 to ``make room for an exciting, NEW, experience.'' (the second phase of The Wizarding World Of Harry Potter.) severe backlash followed after the announcement. The attraction officially closed on January 2, 2012 at 9:00 pm with Michael Skipper aka ``Skip'' giving the final voyage to the last lucky group of 48 guests. By the next morning, the entire Amity area was walled off and completely demolished in the following months. The hanging shark statue from the town square remains as a tribute to the ride and can be found in the Fisherman's Wharf area of the San Francisco section of the park. The attraction remains open at Universal Studios Japan as well as the original tram stop at Universal Studios Hollywood.", "question": "is the jaws ride at universal orlando closed"}
1|0|1|copa|0|1|1|0|1|1
{"choice1": "The toilet filled with water.", "choice2": "Water flowed from the spout.", "idx": 0, "label": 1, "premise": "The man turned on the faucet.", "question": "effect"}
RND2
"choice1": "toilet filled with water The. ", "choice2": "flowed from Water the spout. ", "idx": 0, "label": 1, "premise": "man the on The turned faucet. "

16|1|0|copa|0|1|1|0|0|0
{"choice1": "The courtroom broke into uproar.", "choice2": "The jury announced its verdict.", "idx": 15, "label": 0, "premise": "The judge pounded the gavel.", "question": "cause"}
42|1|0|copa|0|0|1|1|0|0
{"choice1": "The soda fizzed.", "choice2": "The soda leaked out.", "idx": 41, "label": 0, "premise": "I twisted the cap off the soda bottle.", "question": "effect"}
22|1|1|mrpc|1|0|1|0|0|1
{"idx": 195, "label": "1", "text_a": "Last month Intel raised its revenue guidance for the quarter to between $ 7.6 billion and $ 7.8 billion .", "text_b": "At the end of the second quarter , Intel initially predicted sales of between $ 6.9 billion and $ 7.5 billion ."}
247|1|1|qqp|1|1|0|0|0|0
{"idx": 246, "label": "1", "text_a": "How safe are the neighborhoods of Boston?", "text_b": "How safe is Boston?"}
864|0|0|sst|1|0|1|0|1|1
{"idx": 863, "label": "0", "text": "while it 's genuinely cool to hear characters talk about early rap records ( sugar hill gang , etc. ) , the constant referencing of hip-hop arcana can alienate even the savviest audiences . "}

878|1|0|winogrande|1|0|0|0|0|1
"sentence": "I hated the project this year compared to the essay last year, because the _ required more work.", 
"option1": "project", 
"option2": "essay", "answer": "1"
RND2 "because work last I to required the the _ project essay more year compared hated this the year,"

814|0|0|winogrande|1|1|0|1|0|0
{"qID": "3JU8CV4BRNQ92SYBMY4NF7TCJ6SOPD-1", "sentence": "Michael had less money than Samuel did because _ liked to shop and spend money too much.", "option1": "Michael", "option2": "Samuel", "answer": "1"}

## BASH
for NUM in 1 2 3 4 5 6
do
    for TASK in cb copa boolq multirc winogrande
    do
    python jiant/proj/main/runscript.py \
        run \
        --do_val \
        --hf_pretrained_model_name_or_path roberta-base \
        --model_load_mode all \
        --model_config_path /network/scratch/x/xuanda.chen/models/roberta-base/model/config.json \
        --model_path /network/scratch/x/xuanda.chen/tasks/runs/${TASK}/best_model.p \
        --jiant_task_container_config_path /network/scratch/x/xuanda.chen/rnd/rnd${NUM}/run_configs/${TASK}_run_config.json \
        --output_dir /network/scratch/x/xuanda.chen/rnd/rnd${NUM}/runs/${TASK} \
        --write_val_preds \
        --seed 42
    done
done

python jiant/proj/main/runscript.py \
    run \
    --do_val \
    --hf_pretrained_model_name_or_path roberta-base \
    --model_load_mode all \
    --model_config_path /network/scratch/x/xuanda.chen/models/roberta-base/model/config.json \
    --model_path /network/scratch/x/xuanda.chen/tasks/runs/cb/best_model.p \
    --jiant_task_container_config_path /network/scratch/x/xuanda.chen/rnd/rnd1/run_configs/cb_run_config.json \
    --output_dir /network/scratch/x/xuanda.chen/rnd/rnd1/runs/cb \
    --write_val_preds \
    --seed 42


python jiant/proj/main/runscript.py \
    run \
    --do_val \
    --hf_pretrained_model_name_or_path roberta-base \
    --model_load_mode all \
    --model_config_path /Users/xdchen/jiant/models/roberta-base/model/config.json \
    --model_path /Users/xdchen/Downloads/best_model.p \
    --jiant_task_container_config_path /Users/xdchen/jiant/tasks/run_configs/copa_run_config.json \
    --output_dir /Users/xdchen/jiant/tasks/runs/copa \
    --write_val_preds \
    --seed 42

~/ENV/lib/python3.8/site-packages/jiant/proj/main


for i in `find ./ -name 'val_preds.p'`; do RND=${i:2:4}; FNAME=$(basename ${i}); FPATH=$(dirname ${i}); DNAME=$(basename ${FPATH}); cp $i ../output/${RND}-${DNAME}-${FNAME}; done

cp $i ../output/${RND}-${DNAME}-${FNAME}

scp -r mila:/network/scratch/x/xuanda.chen/output /Users/xdchen/Downloads


\begin{table}[ht]
    \small
    \centering
    \begin{tabular}{cccccc}
    \toprule
    \multirow{2}{*}{\textbf{Tasks}}  & \multicolumn{2}{c}{\textbf{Samples}} & \multicolumn{3}{c}{\textbf{Sent Stat}} \\ \cmidrule(l){2-3} \cmidrule(l){4-6}
                & Train  & Val   & \#Len  & \#Num & \#MI    \\ \midrule
    SST-2       & 67349  & 872   & 16.912 & 915   & 1.414   \\ [0.6ex]
    MRPC        & 3668   & 408   & 17.512 & 1162  & 1.261   \\ [0.6ex]
    QQP         & 363846 & 40430 & 10.480 & 87635 & 1.329   \\ [0.6ex]
    RTE         & 2490   & 277   & 16.772 & 943   & 1.395   \\ [0.6ex]
    COPA        & 400    & 100   & 5.991  & 300   & 1.329   \\ [0.6ex]
    BoolQ       & 9427   & 3270  & 19.677 & 18614 & 1.350   \\ [0.6ex]
    WG  & 40398  & 1267  & 17.062 & 1515  & 1.233   \\ [0.6ex]
    \bottomrule
    \end{tabular}
    \caption{Statistics of selected NLU tasks (WG short for WinoGrande). Sentence statistics are calculated on the validation sets.}
    \label{tab1}
    \end{table}


\begin{table}[ht]
    \small
    \centering
    \begin{tabular}{ccccc}
    \toprule
    \multirow{2}{*}{\textbf{Tasks}}  & \multicolumn{2}{c}{\textbf{Samples}} & \multicolumn{2}{c}{\textbf{Sent Stat}} \\ \cmidrule(l){2-3} \cmidrule(l){4-5}
                & Train  & Val   & \#Len  & \#Num    \\ \midrule
    SST-2       & 67349  & 872   & 16.912 & 915      \\ [0.6ex]
    MRPC        & 3668   & 408   & 17.512 & 1162     \\ [0.6ex]
    QQP         & 363846 & 40430 & 10.480 & 87635    \\ [0.6ex]
    RTE         & 2490   & 277   & 16.772 & 943      \\ [0.6ex]
    COPA        & 400    & 100   & 5.991  & 300      \\ [0.6ex]
    BoolQ       & 9427   & 3270  & 19.677 & 18614    \\ [0.6ex]
    WG          & 40398  & 1267  & 17.062 & 1515     \\ [0.6ex]
    \bottomrule
    \end{tabular}
    \caption{Statistics of selected NLU tasks (WG short for WinoGrande). Sentence statistics are calculated on the validation sets.}
    \label{tab1}
    \end{table}