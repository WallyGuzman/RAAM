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
        touch /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/random/"$hidden_size"_"$data_size".out
        python /u/gguzman/CS-394N/Final-Project/RAAM/draam.py --hidden-size $hidden_size --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/gram_4/gram_4_"$data_size".txt > /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/random/"$hidden_size"_"$data_size".out;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        touch /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/sorted/"$hidden_size"_"$data_size".out
        python /u/gguzman/CS-394N/Final-Project/RAAM/draam.py --hidden-size $hidden_size --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/gram_4/gram_4_"$data_size"_sorted.txt > /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/sorted/"$hidden_size"_"$data_size".out;
        done;
done
