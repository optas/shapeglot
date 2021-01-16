"""
The MIT License (MIT)
Originally created sometime in 2018.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import re
from collections import OrderedDict

separate_symbols = (',', '?', '(', ')', '-')
remove_symbols = ('!', '.', ';', '#', '/', ':', '"', '=', '*', '\\', '\'')
replace_symbols = [('&', ' and '), ('n\'t', ' not '), ('\'re', ' are '),
                   ('\'ve', ' have '), ('\'s', ' @s ')]
hack_symbols = [('@s', '\'s')]  # Added/Applied at the end of the ordered dict => can cancel previous substitutions.


def get_naive_tokenizer_substitute_dict():
    substitute_dic = OrderedDict(replace_symbols)

    for s in separate_symbols:
        substitute_dic.update([(s, ' ' + s + ' ')])
    for s in remove_symbols:
        substitute_dic.update([(s, ' ')])
    substitute_dic.update(hack_symbols)
    return substitute_dic


def replace_based_on_dic(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def remove_apostrophes_from_enclosed_words(game_data, verbose=False):
    """ 'standard' will become standard.
    """
    WORD_RE_STR = r"""
    (?:'[a-z]+')       # Words with encapsulated in apostrophes.
    """
    WORD_RE = re.compile(r"(%s)" % WORD_RE_STR, re.VERBOSE | re.I | re.UNICODE)
    
    for i, s in enumerate(game_data.text):
        iterator = WORD_RE.finditer(s)
        new_sentence = list(s)
        for match in iterator:
            start, end = match.span()
            new_sentence[start] = ' '
            new_sentence[end-1] = ' '
        new_sentence = "".join(new_sentence)
        if new_sentence != s:
            if verbose:
                print(game_data.text[i])
                print(new_sentence)
            game_data.at[i, 'text'] = new_sentence
    return game_data