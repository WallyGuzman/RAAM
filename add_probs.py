import argparse
import re

def main(args):
    in_file = open(args.grammar_file, "r")

    for line in in_file:
        init = line.split('-> ')[0]
        elements = line.split('-> ')[1]
        elements = elements.split(' | ')
        prob = 1.0 / len(elements)
        out = init + '-> '
        for iter, element in enumerate(elements):
            if iter == len(elements) - 1:
                elements[iter] = element.split('\n')[0]
                out += elements[iter] + ' [' + ('{0:.' + str(args.num_dec) + 'f}').format(prob) + ']'
                #print(elements[iter])
            else:
                out += elements[iter] + ' [' + ('{0:.' + str(args.num_dec) + 'f}').format(prob) + '] |'

        print(out)

    in_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description='draam.py')

    parser.add_argument('grammar_file', type=str, help='grammar file name')
    parser.add_argument('num_dec', type=str, help='tolerance')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args)