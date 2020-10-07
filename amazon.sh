#!/bin/bash
set -e 

current_path=$PWD
export HUMAN_FILE=~/workspace/NMT/transformer/amazon/human.txt
export SOURCE_FILE=~/workspace/NMT/transformer/amazon/sentiment.test
export PPL_PATH=~/workspace/NMT/transformer/srilm
#export BLEU_PATH=~/workspace/NMT/related_work/controllable-text-attribute-transfer/outputs
export ACC_PATH=~/workspace/NMT/transformer

if [ $# -eq 1 ]
then
    name=$1
fi


#acc
cd $ACC_PATH
x=$(python amazonacc.py --path1 $name)
echo "file name:$name"
echo "Accuracy:$x"

#x=$(python acc.py --path1 $name)
#echo "file name:$name"
#echo "yelp Accuracy:$x"
#BLEU
#cd $BLEU_PATH
cd $current_path
x=$(python multi-bleu.py -hyp $name -ref $HUMAN_FILE)
output_array=($x)
bleu=${output_array[2]}
echo "BLEU:$bleu"
x=$(python multi-bleu.py -hyp $name -ref $SOURCE_FILE)
output_array=($x)
bleu_2=${output_array[2]}
echo "self-BLEU:$bleu_2"

#PPL
cd $PPL_PATH
PATH=/home/heejinking/workspace/NMT/transformer/srilm/bin/i686-m64:$PATH
export PATH
x=$(ngram -ppl $name -lm 3_gram.amazon -order 3 -debug 2)
#an=$(grep -wns 'file' $x -A 13)

output_array=($x)
ppl_v=${output_array[-3]}
echo "PPL:$ppl_v"

