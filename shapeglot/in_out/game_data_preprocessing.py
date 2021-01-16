"""
Released 2019.
@author: Achlioptas Panos
"""

import numpy as np
import pandas as pd
import warnings

from collections import defaultdict
from collections import Counter

from shapeglot.simple_utils import unique_rows
from shapeglot.language.constants import dialogue_symbol, end_of_sentence_symbol,\
    unknown_symbol, special_symbols
from shapeglot.language.helper import substitute_num_to_alpha
from shapeglot.language.tokenizers import replace_based_on_dic, get_naive_tokenizer_substitute_dict,\
    remove_apostrophes_from_enclosed_words
from shapeglot.language.parts_of_speech import transform_tokens_with_super_comparatives

try:
    import spacy
    from spacy.lang.en import English
    nlp = English()
except:
    pass

# Version of pandas installed.
PV = int(pd.__version__.split('.')[1][0])
if PV not in [1, 2]:
    raise ValueError('Not tested pandas version.')


def preprocess_geometry(game_file, specific_sn_models=None):
    """ Converts input csv names of geometric objects into appropriate integers.
        Input:
            game_file: csv following naming established conventions.
            specific_sn_models: if None, the geometries found in the game_file are the universe
            of geometries, i.e. the mapped integers only describe them.
            Else, it is a numpy array, containing the names of the (shape-net) models that will
            make the universe.
        Output:
            game_data (Panda frame)
            sn_model_id_to_integer (dict).
    """
    geometry_tags = ['chair_a', 'chair_b', 'chair_c']
    target_tag = ['target_chair']
    other_tags = ['text', 'game_id', 'trial_num', 'context_condition', 'correct', 'chat_time']
    all_tags = geometry_tags + target_tag + other_tags
    game_data = pd.read_csv(game_file, usecols=all_tags)

    if specific_sn_models is None:
        existing_geo = unique_referenced_geometry_ids(game_data, geometry_tags)
    else:
        existing_geo = np.load(specific_sn_models, allow_pickle=True)  # allow_pickle is added for new np versions.
        existing_geo = existing_geo[list(existing_geo.keys())[0]]
        existing_geo = np.unique(existing_geo)  # To sort them.

    sn_model_to_integer = dict(zip(existing_geo, range(len(existing_geo))))

    # Convert shape-net-ids to integers, e.g., far_980_a_935093c683edbf2087946594df4e196c to int.
    for item in geometry_tags:
        game_data[item] = game_data[item].apply(lambda x: sn_model_to_integer[x])

    return game_data, sn_model_to_integer


def preprocess_language(game_data, spell_corrector=None, replace_rare=0, tokenizer='naive',
                        do_compar_superlative=False, merge_diags=True):
    # Lower-case all text.
    game_data['text'] = game_data['text'].apply(str.lower)
    # Merge dialogues.
    if merge_diags:
        game_data = group_dialogues(game_data, dialogue_symbol)    
    # Tokenize sentences to words.
    game_data = tokenize_game_text(game_data, tokenizer)
    # Spell check.
    if spell_corrector is not None:
        game_data = apply_spelling_corrections(game_data, spell_corrector)
    # Break down -est/-er of superla/compara-tives
    if do_compar_superlative:
        game_data.text = transform_tokens_with_super_comparatives(game_data.text)
    # Replace rare words with <UKN>.
    if replace_rare > 0:
        game_data = replace_rare_words(game_data, replace_rare)
    # Map each word to an integer.
    game_data, word_to_int = convert_words_to_int(game_data)
    return game_data, word_to_int


def make_word_to_int_dict(game_data):
    cnt = word_occurence(game_data)
    for sw in special_symbols:
        if sw in cnt:
            cnt.pop(sw)
    
    all_words = sorted(cnt.keys())          # Sort non-special words.
    for sw in sorted(special_symbols):      # Put special words first.
        if sw != end_of_sentence_symbol:
            all_words.insert(0, sw)

    all_words.insert(0, end_of_sentence_symbol)  # EOS is mapped to 0.
    word_to_int = dict(zip(all_words, range(len(all_words))))
    return word_to_int


def word_occurence(game_data):
    all_words = []
    for sentence in game_data['text']:
        all_words.extend(sentence)
    cnt = Counter(all_words)
    return cnt


def convert_words_to_int(game_data):    
    word_to_int = make_word_to_int_dict(game_data)
    for i, sentence in enumerate(game_data['text']):
        new_sentence = []
        for word in sentence:
            new_sentence.append(word_to_int[word])
        
        if PV == 1:            
            game_data['text'][i] = new_sentence
        elif PV == 2:            
            game_data.at[i, 'text'] = new_sentence
            
    return game_data, word_to_int


def word_occurrence(game_data):
    all_words = []
    for sentence in game_data['text']:
        all_words.extend(sentence)
    cnt = Counter(all_words)
    return cnt


def replace_rare_words(game_data, thres):
    cnt = word_occurrence(game_data)
    black_list = set()
    for key in cnt:
        if cnt[key] <= thres:
            black_list.add(key)
    print('# Rare words:', len(black_list))
    game_data = replace_blacklisted_words(game_data, black_list)
    return game_data
    
    
def replace_blacklisted_words(game_data, black_list, verbose=False):
    ruined_sentences = list()
    for i, sentence in enumerate(game_data['text']):
        new_sentence = []
        for word in sentence:
            if word not in black_list:
                new_sentence.append(word)
            else:
                new_sentence.append(unknown_symbol)

        ok_sentence = False
        for word in new_sentence:  # If the resulting sentence has at least 1 not-special work is ok.
            if word not in special_symbols:
                ok_sentence = True

        if PV == 1:
            game_data['text'][i] = new_sentence
        elif PV == 2:
            game_data.at[i, 'text'] = new_sentence

        if not ok_sentence:
            ruined_sentences.append(i)

    print('Ruined sentences(contain only rare words):', len(ruined_sentences))
    if verbose:
        for r in ruined_sentences:
            print(game_data['text'][r])
    
    white_mask = np.ones(len(game_data), dtype=np.bool)
    white_mask[np.array(ruined_sentences)] = False    
    game_data = game_data[white_mask]   
    game_data = game_data.reset_index(drop=True)
    
    return game_data


def apply_spelling_corrections(game_data, corrector, verbose=False):
    text = game_data['text']
    new_text = []
    for sentence in text:
        corrected = False
        new_sentence = []    
        for word in sentence:
            if word not in corrector:
                new_sentence.append(word)
            else:
                corrected = True
                proposal = corrector[word].split()  # Some times we replace with many words.
                new_sentence.extend(proposal)
        
        new_text.append(new_sentence)
        
        if verbose and corrected:
            print(sentence)
            print(new_sentence)
   
    game_data['text'] = new_text    
    
    # Assertion
    for sentence in game_data['text']:    
        for word in sentence:
            if word in corrector:
                assert(False)
                
    return game_data


def unique_referenced_geometry_ids(game_data, geometry_tags):
    ''' returns all shape-net ids that were exposed in the game.
    '''
    all_sn_references = []
    for tag in geometry_tags:
        # geometry strings look like 'far_980_a_935093c683edbf2087946594df4e196c'
        sn_ids_of_tag = np.array(game_data[tag].apply(lambda x: x.split('_')[-1]))
        all_sn_references.append(sn_ids_of_tag)
    return np.unique(all_sn_references)


def convert_target_id_to_int(game_data, target_tag, geo_tags):
    all_locs = []
    for g_type in geo_tags:
        loc_found = np.where(game_data[target_tag] == game_data[g_type])[0]
        all_locs.append(loc_found)

    if not np.all(sorted(np.hstack(all_locs)) == np.arange(len(game_data))):
        raise ValueError('Some target strings do not coincide with ANY of their corresponding triplet.')

    for i, l in enumerate(all_locs):
        if PV == 1:
            game_data[target_tag].loc[l] = i        
        else:
            game_data.at[l, target_tag] = i


def group_dialogues(game_data, dia_symbol, game_col='game_id', trial_col='trial_num',
                    text_col='text', time_col='chat_time'):
    """ Join the text that is part of a dialogue.
    Input:
        dia_symbol: symbol used to join consecutive dialogue messages.
        game_col: name of column uniquely specifying a game (two humans playing)
        trial_col: name of column specifying the trial (triplet) of the game
        text_col: name of column that holds the text
        time_col: name of column that has time of emission of each message
    """
    grouper = defaultdict(list)
    time_stamps = defaultdict(list)
    
    # Fist break into equiv. classes
    for row in range(len(game_data)):    
        gid = game_data.loc[row, game_col]
        tid = game_data.loc[row, trial_col]
        grouper[(gid, tid)].append(row)
        time_stamps[(gid, tid)].append(float(game_data.loc[row, time_col]))
    
    # Link dialogues & keep only ONE.
    dia_sym = ' ' + dia_symbol + ' '
    keep = []
    for key, val in grouper.items():
        
        if len(val) == 1:          # No dialogue.
            keep.append(val[0])
        else:                      # Dialogue: sort according to time of message emision.
            assert(len(val) > 1)            
            dia_text = []
            sidx = np.argsort(np.array(time_stamps[key]))       
            sval = np.array(val)[sidx]            
            
            for v in sval:
                dia_text.append(game_data.at[v, text_col])
                
            game_data.at[sval[0], text_col] = dia_sym.join(dia_text)  # join all msg
            keep.append(sval[0])                                      # keep only one row.
    game_data = game_data.loc[keep, :]
    game_data = game_data.reset_index(drop=True)
    return game_data


def tokenize_game_text(game_data, tokenizer):
    # Use spacy tokenizer, ignoring <DIA> symbol and white space.
    if tokenizer == 'spacy':
        special_case = [{'ORTH': u'<DIA>'}]
        nlp.tokenizer.add_special_case(u'<DIA>', special_case)
        docs = nlp.pipe([unicode(s, 'utf-8') for s in game_data['text']], batch_size=50)
        tokens = [[t.text for t in d if not t.is_space] for d in docs]
        game_data['original_text'] = [t for t in game_data['text']]
        game_data['tokenized_text'] = [t for t in tokens]
        game_data['text'] = [t for t in tokens]            
    elif tokenizer == 'naive':
        game_data = remove_apostrophes_from_enclosed_words(game_data)
        text = np.array(game_data['text'], dtype=object)
        substitute_dict = get_naive_tokenizer_substitute_dict()
        killed_sentence = False
        for i, sentence in enumerate(text):
            sen = replace_based_on_dic(sentence, substitute_dict)
            tokens = sen.split()
            tokens = substitute_num_to_alpha(tokens)         # map '1-10' to 'one-ten'.
            text[i] = tokens
            if len(tokens) < 1:
                killed_sentence = True
        
        if killed_sentence:
            warnings.warn('Sentence(s) have already killed by the end of tokenization process.')
                
        game_data['text'] = text        
    else:
        raise ValueError('Unknown tokenizer', tokenizer)
    
    return game_data


def condition_mask_of_data(game_data, only_correct=True, geometry_condition=None):
    """
    only_correct (if True): mask will be 1 in location iff listener predicted correctly.
    geometry_condition (string): 'far' or 'close'
    """
    c1 = np.array(game_data.correct)
    if not only_correct:
        c1 = np.ones_like(c1, dtype=np.bool)
    c2 = np.ones_like(c1, dtype=np.bool)
    if geometry_condition is not None:
        c2 = game_data.context_condition == geometry_condition
    mask = np.logical_and(c1, c2)
    return mask


def basic_game_statistics(game_data, word_to_int):
    n_utter = len(game_data)
    print('# Utterances', n_utter)
    
    text = game_data['text']
    diags = 0
    len_of_utter = 0
    n_diag_syms = 0
    for sentence in text:
        len_of_utter += len(sentence)
        if word_to_int[dialogue_symbol] in sentence:
            diags += 1
            n_diag_syms += sentence.count(word_to_int[dialogue_symbol])
            
    len_of_utter -= n_diag_syms

    print('# dialogues', diags)
    print('Average utterance length', len_of_utter / float(n_utter))

    c = game_data['context_condition'] == 'easy'
    print('# Easy triplets', np.sum(c))
    print('Human precision in easy', np.sum(np.logical_and(c,  game_data['correct'])) / float(np.sum(c)))

    c = game_data['context_condition'] == 'hard'
    print('# Hard triplets', np.sum(c))
    print('Human precision in hard', np.sum(np.logical_and(c,  game_data['correct'])) / float(np.sum(c)))
    
    a = np.array(game_data['chair_a'])
    b = np.array(game_data['chair_b'])
    c = np.array(game_data['chair_c'])
    all_triplets = np.vstack((a, b, c)).T
    print('Unique chairs: ', len(np.unique(all_triplets)))
    print('Unique triplets (ignoring target):', len(unique_rows(all_triplets, True)))
