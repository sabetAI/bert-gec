ERRANT_DIR=~/errant/errant/commands
EVAL_FILE=$1
SRC=$EVAL_FILE
TRG=${EVAL_FILE%src}trg
HYP=${EVAL_FILE%src}pred
HYP_M2=$HYP.m2
REF_M2=${EVAL_FILE%src}ref.m2
EXPERIMENT=$2
CHECKPOINT=$3
OUT=${EVAL_FILE%src}score
echo $SRC
echo $TRG
echo $HYP
echo $HYP_M2
echo $REF_M2
echo $REF_M2
echo $OUT

python3 $ERRANT_DIR/parallel_to_m2.py -orig $SRC -cor $HYP -out $HYP_M2 -tok
python3 $ERRANT_DIR/parallel_to_m2.py -orig $SRC -cor $TRG -out $REF_M2 -tok
python3 $ERRANT_DIR/compare_m2.py -ref $REF_M2 -hyp $HYP_M2 -v > $OUT

## echo $EVAL_FILE
## wc -l $EVAL_FILE
tail -6 $OUT
