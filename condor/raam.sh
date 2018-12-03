#!/usr/bin/env bash
#
# shell script to execute Python jobs in vanilla universe
#

HIDDEN_SIZES=(300 600 1200 2400 4800 9600)
DATA_SIZES=(125000 250000 500000 1000000)
# NOTE: Testing with DRAAM only, for now

# Test on random data
for hidden_size in ${HIDDEN_SIZES[*]}; do
    for data_size in ${DATA_SIZES[*]}; do
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        touch ../logs/gram_4/random/"$data_size"_"$hidden_size".out;
        python ../draam.py --hidden-size $hidden_size --training-file ../data/gram_4/gram_4_"$data_size".txt > ../logs/gram_4/random/"$hidden_size"_"$data_size".out;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        touch ../logs/gram_4/sorted/"$data_size"_"$hidden_size".out;
        python ../draam.py --hidden-size $hidden_size --training-file ../data/gram_4/gram_4_"$data_size"_sorted.txt > ../logs/gram_4/sorted/"$hidden_size"_"$data_size".out;
        done;
done
