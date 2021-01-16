import numpy as np
from collections import defaultdict, Counter

from shapeglot.in_out.numpy_dataset import NumpyDataset
from shapeglot.simple_utils import sort_and_count_len_of_dict_values
from shapeglot.in_out.game_data_preprocessing import condition_mask_of_data
from shapeglot.in_out.geometry import group_target_geometries, group_geometries,\
    shuffle_game_geometries, convert_labels_to_one_hot


def make_dataset_for_rnn_based_model(game_data, splits, max_seq_len, only_correct=False,
                                     drop_too_long=True, geo_condition=None,
                                     unique_test_geo=False,
                                     replace_not_in_train=False, seed=None,
                                     bias_train=False):
    """Packages data into a dict with NumpyDataset.

    Input:
        splits: list with train-test-val percentages.
        max_seq_len: (int) utterances will be padded with <EOS> after max_seq_len.
        drop_too_long: (bool) if True, utterances longer than max_seq_len will be ignored.
            This will result in utterances with max_seq_len + 1 length, all ending with zero on the last position.
        only_correct: (bool) if True, utterances which were not correctly predicted by a human will be ignored.
        unique_test_geo: (bool) if True, the train/test/val have distinct target geometries.

    Note: works only with chair-data.
    """

    geo_ids, labels, padded_text, seq_len, mask = prepare_rnn_dataset(game_data, max_seq_len,
                                                                      only_correct, drop_too_long,
                                                                      geo_condition, seed)

    result = dict()

    if unique_test_geo:
        split_ids = split_indices_with_unseen_target_geo_in_test(splits, geo_ids, labels,
                                                                 bias_train=bias_train,
                                                                 seed=seed)
    else:
        split_ids = split_indices_with_seen_target_geo_in_test_strict(splits, geo_ids, labels,
                                                                      seed=seed, debug=True)

    for s in split_ids.keys():
        rids = split_ids[s]

        result[s] = NumpyDataset([geo_ids[rids], labels[rids], padded_text[rids]],
                                 ['in_geo', 'target', 'text'], init_shuffle=False)

    if replace_not_in_train:
        replace_tokens_not_in_train(result)

    return result, split_ids, seq_len, mask


def prepare_rnn_dataset(game_data, max_seq_len, only_correct=False, drop_too_long=True,
                        geo_condition=None, seed=None):
    """ Helper function for ``` make_dataset_for_rnn_based_model() ```. Read notes there.
    """

    # Labels as one-hot, shuffled geometries, padded-text, extract conditioning mask.
    labels = convert_labels_to_one_hot(game_data['target_chair'])
    geo_ids = np.array(game_data[['chair_a', 'chair_b', 'chair_c']], dtype=np.int32)
    geo_ids, labels = shuffle_game_geometries(geo_ids, labels, random_seed=seed)
    padded_text, seq_len = pad_text_symbols_with_zeros(game_data['text'], max_seq_len, force_zero_end=True)
    mask = condition_mask_of_data(game_data, only_correct=only_correct, geometry_condition=geo_condition)

    if drop_too_long:
        short_mask = np.array(game_data.text.apply(lambda x: len(x)) <= max_seq_len, dtype=np.bool)
        mask = np.logical_and(mask, short_mask)

    # Apply mask.
    geo_ids = geo_ids[mask]
    labels = labels[mask]
    padded_text = padded_text[mask]
    seq_len = seq_len[mask]

    return geo_ids, labels, padded_text, seq_len, mask


def pad_text_symbols_with_zeros(text, max_seq_len, dtype=np.float32, force_zero_end=False):
    """
    force_zero_end (bool) if True every sequence will end with zero, alternatively,
        sequences with equal or more elements than max_seq_len will end with the element at that max_seq_len position.
    """
    text_padded = []
    seq_len = []
    
    if force_zero_end:
        last_slot = 1        
    else:
        last_slot = 0
    
    for sentence in text:
        pad_many = max_seq_len - len(sentence) + last_slot
        if pad_many > 0:
            text_padded.append(np.pad(sentence, (0, pad_many), 'constant', constant_values=0))
            seq_len.append(len(sentence))
        else:
            keep_same = min(max_seq_len, len(sentence))
            kept_text = sentence[:keep_same]
            if force_zero_end:
                kept_text.append(0)
            text_padded.append(kept_text)
            seq_len.append(keep_same)
    
    text_padded = np.array(text_padded, dtype=dtype)
    seq_len = np.array(seq_len, dtype)
    return text_padded, seq_len


def replace_tokens_not_in_train(datasets):
    """ replaces them with <UNK>.
        Input:
        datasets: (dict) with NumpyDataset for 'train/test/val'.
    """
    train_tokens = set(np.unique(datasets['train'].text).astype(int))
    unk_int = 1  # TODO-> needs to be taken from word_to_int
    for s in ['test', 'val']:
        for i in np.nditer(datasets[s].text, op_flags=['readwrite']): 
            if int(i) not in train_tokens:
                i[...] = unk_int

                
def bring_target_last(datasets):
    for dataset in datasets.itervalues():
        target_locs = np.argmax(dataset.target, axis=1)
        assert(dataset.n_examples == len(target_locs))
        for row in range(dataset.n_examples):
            loc = target_locs[row] # target's original location.
            if loc != 2:
                # Swap geo-ids.
                temp = dataset.in_geo[row, 2]  # distractor at last loc.
                dataset.in_geo[row, 2] = dataset.in_geo[row, loc]
                dataset.in_geo[row, loc] = temp
                # Swap target-label.
                temp = dataset.target[row, 2]
                dataset.target[row, 2] = dataset.target[row, loc]
                dataset.target[row, loc] = temp
                
        assert(np.all(np.argmax(dataset.target, axis=1) == 2))


def split_indices_with_unseen_target_geo_in_test(loads, geo_ids, labels, 
                                                 bias_train=False, seed=None,
                                                 debug=True):
    """ Args:
            loads: (list) train-test-val split percentages strictly positive and must sum to 1.0. 
            geo_ids: (n x 3) triplets of geo_ids for n game-interactions.
            labels:  (n x 3) indicators of target geometries.  
            bias_train: (boolean) if True, then the training examples will consists of targets with 
        more super-vised utterances than the test/val.
    """
    train_per, test_per, val_per = loads
    if np.sum(loads) != 1.0:
        raise ValueError('train-test-val split must sum to 1.0')
            
    target_classes = group_target_geometries(geo_ids, labels)
    
    # Count number of utterances associated with each target geometry.
    # (To push triplets with "a lot" of utterances into trainining.)
    sorted_target_classes, lengths = sort_and_count_len_of_dict_values(target_classes)
    sorted_target_classes = sorted_target_classes.astype(np.int32)
    
    if bias_train:
        p = lengths / lengths.sum()
    else:
        p = None
    
    if seed is not None:
        np.random.seed(seed)
    
    n_targets = len(target_classes)    
    train_size = int(np.ceil(train_per * n_targets))
    rest_size = n_targets - train_size
    test_size = int(n_targets * test_per)
    val_size = rest_size - test_size
    assert(val_size + train_size + test_size == n_targets)
    train_ids = np.random.choice(sorted_target_classes, train_size, replace=False, p=p)
    rest_ids = np.setdiff1d(sorted_target_classes, train_ids)    
    test_ids = np.random.choice(rest_ids, test_size, replace=False, p=None)
    val_ids = np.setdiff1d(rest_ids, test_ids)
    
    # Back from target geo-ids to rows.
    res = {}
    for s, ids in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
        rows = []
        for i in ids:
            rows.extend(target_classes[i])    
        res[s] = np.array(rows)
        
    if debug:
        set_geos = defaultdict(list)

        for s in ['train', 'test', 'val']:
            for i, l in zip(geo_ids[res[s]], labels[res[s]]):
                set_geos[s].append(i[np.where(l)[0]][0])
            set_geos[s] = set(set_geos[s])
    
        assert (set_geos['train'].isdisjoint(set_geos['test']) and
                set_geos['val'].isdisjoint(set_geos['test']) and 
                set_geos['train'].isdisjoint(set_geos['val']))

        print ('unique geometries in train/test/val',
        len(set_geos['train']), len(set_geos['test']), len(set_geos['val']))

    return res    


def split_indices_with_seen_target_geo_in_test_strict(loads, geo_ids, labels, seed=None, debug=True):
    """ Each triplet (geo/target) is added with (at least one) utterance to the test/val splits.
    """
    
    train_per, test_per, val_per = loads
    if np.sum(loads) != 1.0:
        raise ValueError('train-test-val split must sum to 1.0')
        
    if seed is not None:
        np.random.seed(seed)
        
    eq_classes = group_geometries(geo_ids, labels)
    res = defaultdict(list)
    
    n_total = float(len(geo_ids))
    
    for rows in eq_classes.itervalues():
        if len(rows) < 3: # If for this combination of geos/target have less than 3 examples, add them all to train-set.
            res['train'].extend(rows)
        else:            
            np.random.shuffle(rows)
            
            if len(res['test']) < n_total * test_per:
                res['test'].extend([rows[0]])
            else:
                res['train'].extend([rows[0]])
                
            if len(res['val']) < n_total * val_per:
                res['val'].extend([rows[1]])
            else:
                res['train'].extend([rows[1]])
            
            res['train'].extend(rows[2:])

    for s in ['train', 'test', 'val']:
        res[s] = np.array(res[s])
    
    if debug:
        a = group_geometries(geo_ids[res['test']], labels[res['test']])
        b = group_geometries(geo_ids[res['val']], labels[res['val']])
        c = group_geometries(geo_ids[res['train']], labels[res['train']])
        for i in a:
            assert i in c
        for i in b:
            assert i in c
    return res


def extract_word_bias_vector(in_data, n_words, epsilon=10e-6, dtype=np.float32, eos_int=0):
    word_counts = Counter(np.array(in_data.text).flatten())    
    word_counts[eos_int] = in_data.n_examples
    bias_init_vector = np.zeros(n_words)
    
    for key, value in word_counts.iteritems():
        bias_init_vector[int(key)] = float(value)

    # If your n_words is bigger than the unique words in the in_data,
    # (e.g., can happen if test has tokens not existing in train).
    # Add epsilon to avoid taking log of zero.
    bias_init_vector[bias_init_vector == 0] = epsilon

    #  Log probability
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    
    bias_init_vector = bias_init_vector.astype(dtype)
    return bias_init_vector
