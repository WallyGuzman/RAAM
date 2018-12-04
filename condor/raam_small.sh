#!/usr/bin/env bash
#
# shell script to execute Python jobs in vanilla universe
#

HIDDEN_SIZES=(100 200 300)
DATA_SIZES=(500 1000 2000 4000 8000)
# NOTE: Testing with DRAAM only, for now
DRAAM=/u/gguzman/CS-394N/Final-Project/RAAM/draam.py

for hidden_size in ${HIDDEN_SIZES[*]}; do
    for data_size in ${DATA_SIZES[*]}; do
        echo "Testing nested PPAs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        touch /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/random/"$hidden_size"_"$data_size".out;
        python "$DRAAM" --hidden-size $hidden_size --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_"$data_size".txt > \ 
            /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/small/random/"$hidden_size"_"$data_size".out;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        touch /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/sorted/"$hidden_size"_"$data_size".out;
        python "$DRAAM" --hidden-size $hidden_size --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_"$data_size"_sorted.txt > \ 
            /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/small/sorted/"$hidden_size"_"$data_size".out;

        echo "Testing nested SCs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        touch /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/random/"$hidden_size"_"$data_size"_subord.out;
        python "$DRAAM" --hidden-size $hidden_size --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_subord_"$data_size".txt > \ 
            /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/small/random/"$hidden_size"_"$data_size"_subord.out;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        touch /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/sorted/"$hidden_size"_"$data_size"_subord.out;
        python "$DRAAM" --hidden-size $hidden_size --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_subord_"$data_size"_sorted.txt > \ 
            /u/gguzman/CS-394N/Final-Project/RAAM/logs/gram_4/small/sorted/"$hidden_size"_"$data_size"_subord.out;
        done;
done
