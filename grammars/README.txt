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
    SP -> Sub NP VP
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
    S -> NP VP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with' | 'to' | 'for' | 'from' | 'out'
    E -> '.'
- gram_file_subord_4: Add subordinate clauses to gram_file_4
    S -> NP VP E | NP VP SP E
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    SP -> Sub NP VP
    Sub -> 'where' | 'when'
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with' | 'to' | 'for' | 'from' | 'out'
    E -> '.'
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
