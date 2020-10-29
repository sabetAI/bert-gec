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

if [ -e $PROCESSED_DIR/bin ]; then
    echo Process file already exists
else
    mkdir -p $SUBWORD_NMT
    mkdir -p $PROCESSED_DIR/bin

    $SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $train_src > $PROCESSED_DIR/train.src
    $SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $train_trg > $PROCESSED_DIR/train.trg
    $SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $valid_src > $PROCESSED_DIR/valid.src
    $SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $valid_trg > $PROCESSED_DIR/valid.trg
    $SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $test_src > $PROCESSED_DIR/test.src
    $SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $test_trg > $PROCESSED_DIR/test.trg

    cp $train_src $PROCESSED_DIR/train.bert.src
    cp $valid_src $PROCESSED_DIR/valid.bert.src
    cp $test_src $PROCESSED_DIR/test.bert.src
fi
