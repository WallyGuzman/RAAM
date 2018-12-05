#!/usr/bin/env bash
#
# shell script to execute Python jobs in vanilla universe
#

HIDDEN_SIZES=(100 200 300)
DATA_SIZES=(500 1000 2000 4000 8000)
# NOTE: Testing with one_hot_raam only, for now
DRAAM=/u/gguzman/CS-394N/Final-Project/RAAM/one_hot_raam.py

for hidden_size in ${HIDDEN_SIZES[*]}; do
    for data_size in ${DATA_SIZES[*]}; do
        echo "Testing nested PPAs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --grammar-file /u/gguzman/CS-394N/Final-Project/RAAM/grammars/gram_file_4.txt --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_"$data_size".txt;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --grammar-file /u/gguzman/CS-394N/Final-Project/RAAM/grammars/gram_file_4.txt --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_"$data_size"_sorted.txt;

        echo "Testing nested SCs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --grammar-file /u/gguzman/CS-394N/Final-Project/RAAM/grammars/gram_file_subord_4.txt --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_subord_"$data_size".txt;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --grammar-file /u/gguzman/CS-394N/Final-Project/RAAM/grammars/gram_file_subord_4.txt --training-file /u/gguzman/CS-394N/Final-Project/RAAM/data/small/gram_4_subord_"$data_size"_sorted.txt;
    done;
done
