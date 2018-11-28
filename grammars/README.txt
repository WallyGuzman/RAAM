- gram_file_simple: Starting NLTK demo_grammar without 'slept'
    S -> NP VP
    NP -> Det N
    PP -> P NP
    VP -> 'saw' NP | 'walked' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog'
    P -> 'in' | 'with'
- gram_file_1: Add nested PPs
    S -> NP VP
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog'
    P -> 'in' | 'with'
- gram_file_2: Add three more Ns
    S -> NP VP
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with'
- gram_file_3: Add four more Vs
    S -> NP VP
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with'
- gram_file_4: Add four more Ps
    S -> NP VP
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'in' | 'with' | 'to' | 'for' | 'from' | 'out'
- gram_file_max_prep: Add most common prepositions from here: https://www.thefreedictionary.com/List-of-prepositions.htm
    S -> NP VP
    NP -> Det N
    PP -> P NP | P NP PP
    VP -> 'saw' NP | 'walked' PP | 'heard' NP | 'ran' PP | 'smelled' NP | 'jogged' PP
    Det -> 'the' | 'a'
    N -> 'man' | 'park' | 'dog' | 'cat' | 'house' | 'woman'
    P -> 'about' | 'above' | 'across' | 'after' | 'against' | 'among' | 'around' | 'at' | 'before' | 'behind' | 'below' | 'beside' | 'between' | 'by' | 'down' | 'during' | 'except' | 'for' | 'from' | 'in' | 'inside' | 'into' | 'near' | 'of' | 'off' | 'on' | 'out' | 'over' | 'through' | 'to' | 'toward' | 'under' | 'up' | 'with'
