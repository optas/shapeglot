"""
The MIT License (MIT)
Originally created sometime in 2018.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import numpy as np
from shapeglot.language.constants import special_symbols, end_of_sentence_symbol

digit_to_alpha = {'1': 'one',
                  '2': 'two',
                  '3': 'three',
                  '4': 'four',
                  '5': 'five',
                  '6': 'six',
                  '7': 'seven',
                  '8': 'eight',
                  '9': 'nine',
                  '10': 'ten'}


def substitute_num_to_alpha(tokens):    
    """substitution happens in-place."""
    for i, t in enumerate(tokens):
        if t in digit_to_alpha:
            tokens[i] = digit_to_alpha[t]
    return tokens


def load_glove_pretrained_model(glove_file, dtype=np.float32):
    """ For models downloaded from:
    https://nlp.stanford.edu/projects/glove/
    """
    print("Loading glove model.")
    embedding = dict()
    with open(glove_file, 'r') as f_in:
        for line in f_in:
            s_line = line.split()
            word = s_line[0]
            w_embedding = np.array([float(val) for val in s_line[1:]], dtype=dtype)
            embedding[word] = w_embedding
    print("Done.", len(embedding), " words loaded.")
    return embedding


def ints_to_sentences(words_as_ints, int_to_word, keep_special_syms=True, keep_as_tokens=True):
    """ Given an integer encoding [[1,2,3],...,[12, 22, 23]] return the strings
    based on the mapping `int_to_word`.
    """
    res = []
    for tokens in words_as_ints:
        words = [int_to_word[i] for i in tokens]
        
        # remove trailing <EOS>
        eos_pos = [i for i, p in enumerate(words) if p == end_of_sentence_symbol]        
        if len(eos_pos) > 0:
            stop = eos_pos[0]
        else: 
            stop = len(words)
        
        words = words[:stop]

        if not keep_special_syms:
            temp = []
            for w in words:
                if w not in special_symbols:
                    temp.append(w)
            words = temp

        if keep_as_tokens:
            res.append(words)
        else:
            text = ' '.join(words)
            res.append(text)
    return res


def token_ints_to_sentence(tokens, int_to_word):
    text = [int_to_word[i] for i in tokens]
    text = ' '.join(text)
    stop = text.find(end_of_sentence_symbol)
    if stop == -1:
        stop = len(text)
    text = text[:stop]
    return text
