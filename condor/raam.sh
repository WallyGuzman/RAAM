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
    touch ../logs/gram_1/random/once_$size.out;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/once00 \
        --test-file ../data/gram_1/random/once01 > ../logs/gram_1/random/once_$size.out;
    echo;
    touch ../logs/gram_1/random/twice_$size.out;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/twice00 \
        --test-file ../data/gram_1/random/twice01 > ../logs/gram_1/random/twice_$size.out;
    echo;
    touch ../logs/gram_1/random/thrice_$size.out;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/thrice00 \
        --test-file ../data/gram_1/random/thrice01 > ../logs/gram_1/random/thrice_$size.out;
    echo;
    touch ../logs/gram_1/random/four_$size.out;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/four00 \
        --test-file ../data/gram_1/random/four01 > ../logs/gram_1/random/four_$size.out;
done

# Test on recursive data
for size in ${HIDDEN_SIZES[*]}; do
    python ../draam.py;
    echo;
    echo "Testing with size:" $size "on recursive data";
    echo "python ../draam.py --hidden-size $size";
    echo;
    touch ../logs/gram_1/only/twice_$size.out;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/random/four00 \
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/recurse/recurse_once.txt \
        --test-file ../data/gram_1/only/only_twice.txt > ../logs/gram_1/only/twice_$size.out;
    echo;
    touch ../logs/gram_1/only/thrice_$size.out;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/recurse/recurse_twice.txt \
        --test-file ../data/gram_1/only/only_thrice.txt > ../logs/gram_1/only/thrice_$size.out;
    echo;
    touch ../logs/gram_1/only/four_$size.out;
    python ../draam.py --hidden-size $size --training-file ../data/gram_1/recurse/recurse_thrice.txt \
        --test-file ../data/gram_1/only/only_four.txt > ../logs/gram_1/only/four_$size.out;
done
