EVAL_DIR=$1
EXPERIMENT=$2
CHECKPOINT=$3

files=$EVAL_DIR/*src*

for f in $files
do
	echo "Evaluate $f"
	./evaluate.sh $f $EXPERIMENT $CHECKPOINT
done
