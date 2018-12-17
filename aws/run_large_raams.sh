#!/usr/bin/env bash
#
# shell script to run all experiments
#

set -euo pipefail;

# TODO: word-file arg
HIDDEN_SIZES=(400)
DATA_SIZES=(16000)
DROPOUT_PROBS=(0.5)

AWS_PATH=/home/ubuntu/NN/RAAM

RAAM="$AWS_PATH"/one_hot_raam.py
DRAAM="$AWS_PATH"/draam.py
DRAAM_PLUS="$AWS_PATH"/draam_plus.py

AWS_LOGS="$AWS_PATH"/aws_logs
PARSE_GEN="$AWS_PATH"/parse_gen.py
PPA_GRAM="$AWS_PATH"/grammars/gram_max.txt
SC_GRAM="$AWS_PATH"/grammars/gram_max_subord.txt
AWS_DATA="$AWS_PATH"/aws_data

# Log directories
mkdir -p "$AWS_LOGS"/{raam,draam}/{random,recursive}/{PPA,SC};
mkdir -p "$AWS_LOGS"/draam_plus/{random,recursive}/{PPA,SC}/{tanh,relu};

for data_size in ${DATA_SIZES[*]}; do
    # Generate data according to PPA grammar
    python "$PARSE_GEN" -g "$PPA_GRAM" -n "$data_size" > "$AWS_DATA"/gram_4.txt;
    awk '{ print (NF-1), $0 }' "$AWS_DATA"/gram_4.txt | sort -n | cut -d" " -f 2- > "$AWS_DATA"/gram_4_sorted.txt;

    # Generate data according to SC grammar
    python "$PARSE_GEN" -g "$SC_GRAM" -n "$data_size" > "$AWS_DATA"/gram_4_subord.txt;
    awk '{ print (NF-1), $0 }' "$AWS_DATA"/gram_4_subord.txt | sort -n | cut -d" " -f 2- > "$AWS_DATA"/gram_4_subord_sorted.txt;

    for hidden_size in ${HIDDEN_SIZES[*]}; do
	# Run RAAM
	echo "BEGIN RAAM";
        echo "Testing nested PPAs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        python "$RAAM" --hidden-size $hidden_size --grammar-file "$PPA_GRAM" --word-file PPA_random_raam.npy --report-test --training-file "$AWS_DATA"/gram_4.txt > "$AWS_LOGS"/raam/random/PPA/"$data_size"_"$hidden_size".log;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        python "$RAAM" --hidden-size $hidden_size --grammar-file "$PPA_GRAM" --word-file PPA_recursive_raam.npy --report-test --training-file "$AWS_DATA"/gram_4_sorted.txt > "$AWS_LOGS"/raam/recursive/PPA/"$data_size"_"$hidden_size".log;

        echo "Testing nested SCs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        python "$RAAM" --hidden-size $hidden_size --grammar-file "$SC_GRAM" --word-file SC_random_raam.npy --report-test --training-file "$AWS_DATA"/gram_4_subord.txt > "$AWS_LOGS"/raam/random/SC/"$data_size"_"$hidden_size".log;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        python "$RAAM" --hidden-size $hidden_size --grammar-file "$SC_GRAM" --word-file SC_recursive_raam.npy --report-test --training-file "$AWS_DATA"/gram_4_subord_sorted.txt > "$AWS_LOGS"/raam/recursive/SC/"$data_size"_"$hidden_size".log;
	echo "END RAAM";

	# Run DRAAM
	echo "BEGIN DRAAM";
        echo "Testing nested PPAs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --word-file PPA_random_draam.npy --report-test --training-file "$AWS_DATA"/gram_4.txt > "$AWS_LOGS"/draam/random/PPA/"$data_size"_"$hidden_size".log;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --word-file PPA_recursive_draam.npy --report-test --training-file "$AWS_DATA"/gram_4_sorted.txt > "$AWS_LOGS"/draam/recursive/PPA/"$data_size"_"$hidden_size".log;

        echo "Testing nested SCs";
        echo "Testing with size:" $hidden_size "on random data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --word-file SC_random_draam.npy --report-test --training-file "$AWS_DATA"/gram_4_subord.txt > "$AWS_LOGS"/draam/random/SC/"$data_size"_"$hidden_size".log;
        echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
        echo;
        python "$DRAAM" --hidden-size $hidden_size --word-file SC_recursive_draam.npy --report-test --training-file "$AWS_DATA"/gram_4_subord_sorted.txt > "$AWS_LOGS"/draam/recursive/SC/"$data_size"_"$hidden_size".log;
	echo "END DRAAM";

	# Run DRAAM+
	echo "BEGIN DRAAM+";
	for prob in ${DROPOUT_PROBS[*]}; do
		echo "BEGIN PROB";
		echo "Testing nested PPAs";
		echo "Testing with size:" $hidden_size "on random data:" $data_size "with activation relu and dropout prob:" "$prob";
		echo;
		python "$DRAAM_PLUS" --keep-prob "$prob" --extra-hidden --activation "relu" --hidden-size $hidden_size --word-file PPA_random_draam_plus.npy --report-test --training-file "$AWS_DATA"/gram_4.txt > "$AWS_LOGS"/draam_plus/random/PPA/relu/"$data_size"_"$hidden_size"_"$prob".log;
		echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
		echo;
		python "$DRAAM_PLUS" --keep-prob "$prob" --extra-hidden --activation "relu" --hidden-size $hidden_size --word-file PPA_recursive_draam_plus.npy --report-test --training-file "$AWS_DATA"/gram_4_sorted.txt > "$AWS_LOGS"/draam_plus/recursive/PPA/relu/"$data_size"_"$hidden_size"_"$prob".log;

		echo "Testing nested SCs";
		echo "Testing with size:" $hidden_size "on random data:" $data_size;
		echo;
		python "$DRAAM_PLUS" --keep-prob "$prob" --extra-hidden --activation "relu" --hidden-size $hidden_size --word-file SC_random_draam_plus.npy --report-test --training-file "$AWS_DATA"/gram_4_subord.txt > "$AWS_LOGS"/draam_plus/random/SC/relu/"$data_size"_"$hidden_size"_"$prob".log;
		echo "Testing with size:" $hidden_size "on recursive data:" $data_size;
		echo;
		python "$DRAAM_PLUS" --keep-prob "$prob" --extra-hidden --activation "relu" --hidden-size $hidden_size --word-file SC_recursive_draam_plus.npy --report-test --training-file "$AWS_DATA"/gram_4_subord_sorted.txt > "$AWS_LOGS"/draam_plus/recursive/SC/relu/"$data_size"_"$hidden_size"_"$prob".log;
		echo "END PROB";
	done;
	echo "END DRAAM+";
    done;
done
