INPATH="/network/scratch/x/xuanda.chen/"
ONPATH="/Users/xdchen/Downloads/eval2/data/"

# for RND in rnd1 rnd2 rnd3 rnd4 rnd5 rnd6
# do
#     mkdir -p "${ONPATH}data-rnd/${RND}/runs"
#     # echo "${ONPATH}data-rnd/${RND}/runs"
#     for TASK in boolq cb copa sst multirc rte qqp mrpc winogrande

#     do
#     mkdir -p "${ONPATH}data-rnd/${RND}/runs/${TASK}"
#     #scp mila:$INPATH"rnd/"$RND"/data"
#     done
# done

# for RND in rnd1 rnd2 rnd3 rnd4 rnd5 rnd6
# do
#     for TASK in winogrande

#     do
#     mkdir -p "${ONPATH}data-rnd/${RND}/runs/${TASK}/winogrande_1.1"
#     done
# done

# CP
INPATH="/network/scratch/x/xuanda.chen/";ONPATH="/Users/xdchen/Downloads/eval2/data/";for RND in rnd1 rnd2 rnd3 rnd4 rnd5 rnd6; do; for TASK in boolq cb copa sst multirc rte qqp mrpc; do; scp mila:"${INPATH}rnd/${RND}/data/${TASK}/score.csv" "${ONPATH}data-rnd/${RND}/data/${TASK}/score.csv"; scp mila:"${INPATH}rnd/${RND}/runs/${TASK}/val_preds.p" "${ONPATH}data-rnd/${RND}/runs/${TASK}/val_preds.p"; done; done

INPATH="/network/scratch/x/xuanda.chen/";ONPATH="/Users/xdchen/Downloads/eval2/data/";for RND in rnd1 rnd2 rnd3 rnd4 rnd5 rnd6;do;for TASK in winogrande;do;scp mila:"${INPATH}rnd/${RND}/data/${TASK}/winogrande_1.1/score.csv" "${ONPATH}data-rnd/${RND}/data/${TASK}/winogrande_1.1/score.csv";scp mila:"${INPATH}rnd/${RND}/runs/${TASK}/val_preds.p" "${ONPATH}data-rnd/${RND}/runs/${TASK}/val_preds.p" ;done;done


for i in `find ./ -name 'score.csv'`; do echo $i; FNAME=$(basename ${i}); FPATH=$(dirname ${i}); ; done
FNAME=$(basename ${i})
FPATH=$(dirname ${i})
DNAME=$(basename ${FNAME})
mv ${i} ${FPATH}/${DNAME}.${FNAME}
done

cpath="~/scratch/output/"

cpath="~/scratch/output/";for i in `find ./ -name 'score.csv'`; do RND=${i:2:4}; FNAME=$(basename ${i}); FPATH=$(dirname ${i}); DNAME=$(basename ${FPATH}); cp $i ${cpath}${RND}-${DNAME}-${FNAME}; done

for i in `find ./ -name 'val_preds.p'`; do RND=${i:2:4}; FNAME=$(basename ${i}); FPATH=$(dirname ${i}); DNAME=$(basename ${FPATH}); cp $i ../output/${RND}-${DNAME}-${FNAME}; done

 cp $i ../output/${RND}-${DNAME}-${FNAME}; done