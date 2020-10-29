input=$1
experiment=$2
beam=5
bert_type=bert-base-cased
seed=2222
checkpoint=$3
output=$4

SUBWORD_NMT=../subword
FAIRSEQ_DIR=../bert-nmt
BPE_MODEL_DIR=../gec-pseudodata/bpe
MODEL_DIR=../model/$bert_type/$experiment
OUTPUT_DIR=$MODEL_DIR/output/$checkpoint
PROCESSED_DIR=../process/$experiment

echo Generating...
python3 -u ${FAIRSEQ_DIR}/interactive.py ${PROCESSED_DIR}/bin \
    --path ${MODEL_DIR}/${checkpoint}.pt \
    --beam ${beam} \
    --nbest ${beam} \
    --buffer-size 1024 \
    --batch-size 32 \
    --log-format simple \
    --remove-bpe \
    --bert-model-name $bert_type \
    < $input > $output
