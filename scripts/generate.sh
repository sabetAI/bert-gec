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

mkdir -p $OUTPUT_DIR

$SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $input > $OUTPUT_DIR/test.bpe.src
python3 -u detok.py $input $OUTPUT_DIR/test.bert.src
paste -d "\n" $OUTPUT_DIR/test.bpe.src $OUTPUT_DIR/test.bert.src > $OUTPUT_DIR/test.cat.src

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
    < $OUTPUT_DIR/test.cat.src > $OUTPUT_DIR/test.nbest.tok

cat $OUTPUT_DIR/test.nbest.tok | grep "^H"  | python3 -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if (i % ${beam} == 0) ]); print(x)" | cut -f3 > $OUTPUT_DIR/test.best.tok
sed -i '$d' $OUTPUT_DIR/test.best.tok
cp $OUTPUT_DIR/test.best.tok $output
