- gram_file_simple: Starting NLTK demo_grammar without 'slept'
    S -> NP VP E
    NP -> Det N
    PP -> P NP
    VP -> 'saw' NP | 'walked' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_subord_simple: Add subordinate clauses to gram_file_simple
    S -> NP VP E | NP VP SP E
    NP -> Det N
    PP -> P NP
    VP -> 'saw' NP | 'walked' PP 
    SP -> Sub NP VP | Sub NP VP SP
    Sub -> 'where' | 'when'
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_1: Add nested PPs
    S -> NP VP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_subord_1: Add subordinate clauses to gram_file_1
    S -> NP VP E | NP VP SP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP 
    SP -> Sub NP VP
    Sub -> 'where' | 'when'
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_2: Add three more Ns
    S -> NP VP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_subord_2: Add subordinate clauses to gram_file_2
    S -> NP VP E | NP VP SP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP
    SP -> Sub NP VP
    Sub -> 'where' | 'when'
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_3: Add four more Vs
    S -> NP VP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_subord_3: Add subordinate clauses to gram_file_3
    S -> NP VP E | NP VP SP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    SP -> Sub NP VP
    Sub -> 'where' | 'when'
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with'
    E -> '.'
- gram_file_4: Add four more Ps
    S -> NP VP E [1.0]
    NP -> Det N [1.0]
    PP -> P NP [0.5] | P NP PP [0.5]
    VP -> 'saw' NP [0.16667] | 'walked' PP [0.16667] | 'heard' NP [0.16667] | 'ran' PP [0.16667] | 'smelled' NP [0.16667] | 'jogged' PP [0.16667]
    Det -> 'the' [0.5] | 'a' [0.5]
    N -> 'man' [0.16667] | 'park' [0.16667] | 'dog' [0.16667] | 'cat' [0.16667] | 'house' [0.16667] | 'woman' [0.16667]
    P -> 'in' [0.16667] | 'with' [0.16667] | 'to' [0.16667] | 'for' [0.16667] | 'from' [0.16667] | 'out' [0.16667]
    E -> '.' [1.0]
- gram_file_subord_4: Add subordinate clauses to gram_file_4
    S -> NP VP E [0.5] | NP VP SP E [0.5]
    NP -> Det N [1.0]
    PP -> P NP [0.5] | P NP PP [0.5]
    VP -> 'saw' NP [0.16667] | 'walked' PP [0.16667] | 'heard' NP [0.16667] | 'ran' PP [0.16667] | 'smelled' NP [0.16667] | 'jogged' PP [0.16667]
    SP -> Sub NP VP [1.0]
    Sub -> 'where' [0.5] | 'when' [0.5]
    Det -> 'the' [0.5] | 'a' [0.5]
    N -> 'man' [0.16667] | 'park' [0.16667] | 'dog' [0.16667] | 'cat' [0.16667] | 'house' [0.16667] | 'woman' [0.16667]
    P -> 'in' [0.16667] | 'with' [0.16667] | 'to' [0.16667] | 'for' [0.16667] | 'from' [0.16667] | 'out' [0.16667]
    E -> '.' [1.0]
- gram_file_max_prep: Add most common prepositions from here: https://www.thefreedictionary.com/List-of-prepositions.htm
    S -> NP VP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'about' | 'above' | 'across' | 'after' | 'against' | 'among' | 'around' | 'at' | 'before' | 'behind' | 'below' | 'beside' | 'between' | 'by' | 'down' | 'during' | 'except' | 'for' | 'from' | 'in' | 'inside' | 'into' | 'near' | 'of' | 'off' | 'on' | 'out' | 'over' | 'through' | 'to' | 'toward' | 'under' | 'up' | 'with'
    E -> '.'
- gram_file_max_prep_subord: Add subordinate clauses to gram_file_max_prep
    S -> NP VP E | NP VP SP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    SP -> Sub NP VP
    Sub -> 'where' | 'when'
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'about' | 'above' | 'across' | 'after' | 'against' | 'among' | 'around' | 'at' | 'before' | 'behind' | 'below' | 'beside' | 'between' | 'by' | 'down' | 'during' | 'except' | 'for' | 'from' | 'in' | 'inside' | 'into' | 'near' | 'of' | 'off' | 'on' | 'out' | 'over' | 'through' | 'to' | 'toward' | 'under' | 'up' | 'with'
    E -> '.'
