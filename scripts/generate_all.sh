#!/bin/bash

folder=$1
model=$2
checkpoint=$3
type=$4
files=$folder/*.$type.src

for file in $files
do	
	echo $file
	output=$file
	output=${output%src}pred
	./generate.sh $file $model $checkpoint $output
done
