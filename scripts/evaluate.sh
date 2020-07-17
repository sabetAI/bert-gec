ERRANT_DIR=~/errant/errant/commands
DATA_DIR=~/data
EVAL_FILE=$1
REF_SRC=$DATA_DIR/$EVAL_FILE.src
REF_TRG=$DATA_DIR/$EVAL_FILE.trg
EXPERIMENT=$2
CHECKPOINT=$3
OUTPUT_DIR=~/bert-gec/model/bert-base-cased/$EXPERIMENT/output/$CHECKPOINT

python3 $ERRANT_DIR/parallel_to_m2.py -orig $REF_SRC -cor $OUTPUT_DIR/test.best.tok -out $OUTPUT_DIR/test.best.m2 -tok
python3 $ERRANT_DIR/parallel_to_m2.py -orig $REF_SRC -cor $REF_TRG -out $OUTPUT_DIR/$EVAL_FILE.m2 -tok
python3 $ERRANT_DIR/compare_m2.py -ref $OUTPUT_DIR/$EVAL_FILE.m2 -hyp $OUTPUT_DIR/test.best.m2 -v > $OUTPUT_DIR/test.best.score

tail -6 $OUTPUT_DIR/test.best.score
