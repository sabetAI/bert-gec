mkdir -p ../data

python3 -u ~/errant/errant/commands/convert_m2_to_parallel.py ../data/wi+locness/m2/ABC.train.gold.bea19.m2 ../data/bea19.train.src ../data/bea19.train.trg
python3 -u ~/errant/errant/commands/convert_m2_to_parallel.py ../data/wi+locness/m2/ABCN.dev.gold.bea19.m2 ../data/bea19.valid.src ../data/bea19.valid.trg
python3 -u ~/errant/errant/commands/convert_m2_to_parallel.py ../data/wi+locness/m2/ABCN.dev.gold.bea19.m2 ../data/bea19.test.src ../data/bea19.test.trg
