#! /usr/bin/env python3
# parse_gen.py

import os
import sys
from nltk.parse.generate import generate
from nltk.grammar import Nonterminal
from nltk import PCFG
from random import choice
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

# Stolen from here:
# https://stackoverflow.com/questions/15009656/how-to-use-nltk-to-generate-sentences-from-an-induced-grammar/15617664
# def generate_sample(grammar, items=["S"]):
#     frags = []
#     if len(items) == 1:
#         if isinstance(items[0], Nonterminal):
#             for prod in grammar.productions(lhs=items[0]):
#                 frags.append(generate_sample(grammar, prod.rhs()))
#         else:
#             frags.append(items[0])
#     else:
#         chosen_expansion = choice(items)
#         frags.append(generate_sample, chosen_expansion)
#     return frags

# Stolen from here:
# https://stackoverflow.com/questions/15009656/how-to-use-nltk-to-generate-sentences-from-an-induced-grammar/15617664 
def generate_sample(grammar, prod, frags):
    if prod in grammar._lhs_index:
        derivations = grammar._lhs_index[prod]
        derivation = choice(derivations)
        for d in derivation._rhs:
            generate_sample(grammar, d, frags)
    elif prod in grammar._rhs_index:
        # terminal
        frags.append(prod)

def generate_sentences(args):
    grammar_string = ""

    with open(args.grammar_file, "r") as gram_file:
        grammar_string = gram_file.read()

    grammar = PCFG.fromstring(grammar_string)

    if args.verbose:
        print(grammar)
        print()

    for _ in range(args.num_sent):
        frags = []
        generate_sample(grammar, grammar.start(), frags)
        yield ' '.join(frags)

if __name__ == "__main__":
    args = parse_args()

    for sentence in generate_sentences(args):
        args.output_file.write(sentence)
        args.output_file.write('\n')
