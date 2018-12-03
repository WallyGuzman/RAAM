#!/usr/bin/env bash
#
# shell script to execute Python jobs in vanilla universe
#

HIDDEN_SIZES=(300 600 1200 2400 4800 9600)
# NOTE: Testing with DRAAM only, for now

# Test on random data
for size in ${HIDDEN_SIZES[*]}; do
    echo "Testing with size:" $size "on random data";
    echo "python ../draam.py --hidden-size $size";
    echo;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/once00 \
        --test-file ../data/gram_1/random/once01 > ../logs/gram_1/random/once_$size.out;
    echo;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/twice00 \
        --test-file ../data/gram_1/random/twice01 > ../logs/gram_1/random/twice_$size.out;
    echo;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/thrice00 \
        --test-file ../data/gram_1/random/thrice01 > ../logs/gram_1/random/thrice_$size.out;
    echo;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/four00 \
        --test-file ../data/gram_1/random/four01 > ../logs/gram_1/random/four_$size.out;
done

# Test on recursive data
# for size in ${HIDDEN_SIZES[*]}; do
#     python ../draam.py;
#     echo;
# done

# Recursion tests: (train in gram_1/recurse, test in gram_1/only)
#     1.  train:  recurse_once.txt,    test:  only_twice.txt
#     2.  train:  recurse_twice.txt,   test:  only_thrice.txt
#     3.  train:  recurse_thrice.txt,  test:  only_four.txt
