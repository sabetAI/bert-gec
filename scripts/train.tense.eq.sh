#!/bin/bash

bert_type=bert-base-cased
seed=2222
gec_model=../pseudo_model/ldc_giga.spell_error.pretrain.checkpoint_last.pt
bert_model=../bert-base-cased
experiment=tense-eq
checkpoint=checkpoint_pretrain
epochs=5
save_interval=10000

SUBWORD_NMT=../subword
FAIRSEQ_DIR=../bert-nmt
BPE_MODEL_DIR=../gec-pseudodata/bpe
TRAIN_DIR=~/filter-data
VALID_DIR=~/gec-data/conll14st-test-data/noalt
VOCAB_DIR=../gec-pseudodata/vocab
PROCESSED_DIR=../process/$experiment
MODEL_DIR=../model/$bert_type/$experiment
LOG_DIR=../model/$bert_type/$experiment/logs

pre_trained_model=../pretrained/ldc_giga.spell_error.pretrain.checkpoint_last.pt

train_src=$TRAIN_DIR/bea.tense.eq.src
train_trg=$TRAIN_DIR/bea.tense.eq.trg
valid_src=$VALID_DIR/official-2014.combined.src 
valid_trg=$VALID_DIR/official-2014.combined.trg
test_src=$VALID_DIR/official-2014.combined.src
test_trg=$VALID_DIR/official-2014.combined.trg

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

    python3 $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang trg \
        --trainpref $PROCESSED_DIR/train \
        --validpref $PROCESSED_DIR/valid \
        --testpref $PROCESSED_DIR/test \
        --destdir $PROCESSED_DIR/bin \
        --srcdict $VOCAB_DIR/dict.src_bpe8000.txt \
        --tgtdict $VOCAB_DIR/dict.trg_bpe8000.txt \
        --workers $cpu_num \
        --bert-model-name $bert_type
fi


mkdir -p $MODEL_DIR

if [ ! -e "$MODEL_DIR/checkpoint_pretrain.pt" ]; then
	echo "Copying pretrain file..."
	cp $pre_trained_model $MODEL_DIR/checkpoint_pretrain.pt
fi

CUDA_VISIBLE_DEVICES=0,1 python3 -u $FAIRSEQ_DIR/train.py $PROCESSED_DIR/bin \
    --tensorboard-logdir $LOG_DIR \
    --save-dir $MODEL_DIR \
    --arch transformer_s2_vaswani_wmt_en_de_big \
    --max-tokens 4096 \
    --optimizer adam \
    --lr 0.00003 \
    -s src \
    -t trg \
    --dropout 0.3 \
    --lr-scheduler reduce_lr_on_plateau \
    --lr-shrink 0.7 \
    --min-lr 1e-06 \
    --warmup-from-nmt \
    --warmup-nmt-file $checkpoint.pt \
    --bert-model-name $bert_model \
    --encoder-bert-dropout \
    --encoder-bert-dropout-ratio 0.3 \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch $epochs \
    --adam-betas '(0.9,0.98)' \
    --log-format simple \
    --save-interval-updates $save_interval \
    --fp16 \
    --seed $seed \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader
