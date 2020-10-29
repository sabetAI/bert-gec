#!/bin/bash
bert_type=bert-base-cased
seed=2222
gec_model=../pseudo_model/ldc_giga.spell_error.pretrain.checkpoint_last.pt
bert_model=../bert-base-cased
experiment=bea-proper
checkpoint=checkpoint_pretrain
epochs=5
save_interval=10000

SUBWORD_NMT=../subword
FAIRSEQ_DIR=../bert-nmt
BPE_MODEL_DIR=../gec-pseudodata/bpe
DATA_DIR=~/gec-data/
VOCAB_DIR=../gec-pseudodata/vocab
PROCESSED_DIR=../process/$experiment
MODEL_DIR=../model/$bert_type/$experiment
LOG_DIR=../model/$bert_type/$experiment/logs

pre_trained_model=../pretrained/ldc_giga.spell_error.pretrain.checkpoint_last.pt

train_src=$DATA_DIR/merged/bea.train.dev.src
train_trg=$DATA_DIR/merged/bea.train.dev.trg
valid_src=$DATA_DIR/conll14st-test-data/noalt/official-2014.combined.src
valid_trg=$DATA_DIR/conll14st-test-data/noalt/official-2014.combined.trg
test_src=$DATA_DIR/conll14st-test-data/noalt/official-2014.combined.src
test_trg=$DATA_DIR/conll14st-test-data/noalt/official-2014.combined.trg

cpu_num=`grep -c ^processor /proc/cpuinfo`

python3 $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang trg \
    --trainpref $PROCESSED_DIR/train \
    --validpref $PROCESSED_DIR/valid \
    --testpref $PROCESSED_DIR/test \
    --destdir $PROCESSED_DIR/bin \
    --srcdict $VOCAB_DIR/dict.src_bpe8000.txt \
    --tgtdict $VOCAB_DIR/dict.trg_bpe8000.txt \
    --workers $cpu_num \
    --bert-model-name $bert_type
