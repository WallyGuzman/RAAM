#! /usr/bin/env python3
# parse_gen.py

import os
import sys
from nltk.parse.generate import generate
from nltk import CFG
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='parse_gen.py')

    parser.add_argument('-n', '--num-sent', type=int, default=10, help='number of sentences')
    parser.add_argument('-d', '--max_depth', type=int, default=4, help='maximum depth')
    parser.add_argument('-g', '--grammar-file', type=str, default='grammars/gram_file_simple.txt', help='grammar file')
    parser.add_argument('-o', '--output-file', type=argparse.FileType('w'), default=sys.stdout, help='generated file')
    parser.add_argument('-v', '--verbose', action="store_true", help="verbose flag")
    
    args = parser.parse_args()
    return args

def generate_sentences(args):
    grammar_string = ""

    with open(args.grammar_file, "r") as gram_file:
        grammar_string = gram_file.read()

    grammar = CFG.fromstring(grammar_string)

    if args.verbose:
        print(grammar)
        print()

    if args.max_depth > 4:
        gen = generate(grammar, depth=args.max_depth)
    else:
        gen = generate(grammar, n=args.num_sent)

    for sentence in gen:
        yield sentence

if __name__ == "__main__":
    args = parse_args()

    for sentence in generate_sentences(args):
        args.output_file.write(' '.join(sentence))
        args.output_file.write('\n')
