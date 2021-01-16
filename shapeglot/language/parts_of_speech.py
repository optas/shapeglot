"""
The MIT License (MIT)
Originally created sometime in 2018.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import nltk

SUPER_COMPAR = ['JJS', 'RBS', 'JJR', 'RBR']

irregular_super_compar = {
    'best': ('good', 'est'), 
    'better': ('good', 'er'),
    'least': ('less',  'est'),
    'farthest': ('far', 'est'), 
    'farther': ('far', 'er'),
    'further': ('far', 'er'),
    'furthest': ('far', 'est'),
    'fatter': ('fat', 'er'), 
    'fattest': ('fat', 'est'),
    'flatter': ('flat', 'er'), 
    'flattest': ('flat', 'est'),
    'slimmer': ('slim', 'er'),
    'slimmest': ('slim', 'est'),
    'bigger': ('big', 'er'),
    'biggest': ('big', 'est'),
    'thinnest': ('thin', 'est'),
    'thinner': ('thin', 'er'),
    'funnest': ('fun', 'est'),
    'littlest': ('little', 'est'),
    'littler': ('little', 'er'),
    }

# close -> closer, thus you can't simply remove -er
adjectives_ending_in_e = {'close', 'simple', 'square', 'strange', 'wide', 'large'}

# nouns ending in -er/-est that are marked as "ives" by POS.
not_super_compar = set(['armrest', 'backrest', 'booster', 'cylinder', 'flower', 'footrest',
                       'further', 'headrest', 'ladder', 'lest', 'lounger', 'rest', 'wither',
                       'scraper', 'slender', 'spider', 'super', 'taper', 'tier', 'est', 'pooper',
                       'answer', 'honer', 'ever', 'viewer', 'everest', 'forrest', 'nest', 'rocker',
                       'seater', ])

super_compar_missed = set(['plushest']) # POS didn't reliaze the are comparatives.


def get_parts_of_speech(token_list):
    """ Given a list of tokenized sentences (strings) returns for
    each token of each sentence the POS.
    """
    res = []
    for tokens in token_list:
        res.append(nltk.pos_tag(tokens))
    return res


def transform_tokens_with_super_comparatives(tokens):
    """
    Note: The result of this function depend (also) on some 'rules' for exceptions that we manually
    curated to improve the nltk POS tagger when applied on shapeglot data (see global symbols in this file).
    """
    res = []
    pos_text = get_parts_of_speech(tokens)
    
    # POS-TAG might loose sometimes a superlative & find it in another sentence.
    # Collect all those found, to make sure you mark them in other sentences too.
    mined_ives = super_compar_missed.copy()
    for pt in pos_text:
        for token in pt:
            if token[1] in SUPER_COMPAR:
                mined_ives.add(token[0])
           
    for pt in pos_text:
        new_tokens = []
        for token in pt:
            word = token[0]
            converted = False
            if token[1] in SUPER_COMPAR or word in mined_ives:
                # better->good/er
                if word in irregular_super_compar:
                    root, ending = irregular_super_compar[word]
                    new_tokens.append(root)
                    new_tokens.append(ending)                    
                    continue
                                
                for ending in ['est', 'er']:
                    if word.endswith(ending) and (word not in not_super_compar):                        
                        # simple->simpler
                        if word[:-(len(ending) - 1)] in adjectives_ending_in_e:
                            root = word[:-(len(ending) - 1)]

                        # fancy->fancier
                        elif word.endswith('i' + ending):
                            root = word.replace('i' + ending, 'y')

                        # tall->taller
                        else:
                            root = word.replace(ending, '')                    
                            
                        new_tokens.append(root)
                        new_tokens.append(ending)
                        converted = True
                        continue
            if not converted:
                new_tokens.append(word)                                    
        res.append(new_tokens)
    return res
