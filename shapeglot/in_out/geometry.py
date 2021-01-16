import numpy as np
from collections import defaultdict

from shapeglot.in_out.helper import snc_category_to_synth_id
from shapeglot.simple_utils import unpickle_data


def vgg_image_features(int_to_sn_model, class_name, vgg_feats_file, dtype=np.float32, python2_to_3=False):
    """ For models in input dictionary load/return pre-trained vgg-features according to the order implied
    by the given dictionary.
    Input:
        int_to_sn_model (dict): describes a set of models-ids and an order (int) for each of them.
        class_name (string), e.g. 'chair'
    """
    vgg_feats = next(unpickle_data(vgg_feats_file, python2_to_3=python2_to_3))
    a_key = next(iter(vgg_feats))
    feat_dim = vgg_feats[a_key].shape
    n_geometries = len(int_to_sn_model)    
    vgg_emb = np.zeros(shape=(n_geometries,) + feat_dim, dtype=dtype)
    syn_id = snc_category_to_synth_id()[class_name]
    
    for i, model_name in int_to_sn_model.items():
        vgg_emb[i] = vgg_feats[syn_id + '_' + model_name]

    return vgg_emb


def pc_ae_features(int_to_sn_model, geo_embedding_file, dtype=np.float32):
    geo_embedding = np.load(geo_embedding_file)
    all_model_ids = geo_embedding['model_ids']
    geo_codes = geo_embedding['latent_codes']
    
    sn_model_to_int = dict(zip(all_model_ids, range(len(all_model_ids))))
    geo_emb_dim = geo_codes.shape[1]
    n_geometries = len(int_to_sn_model)
    ae_emb = np.zeros(shape=(n_geometries, geo_emb_dim), dtype=dtype)

    for i,  model_name in int_to_sn_model.items():
        # ae_emb[i] = geo_codes[sn_model_to_int[model_name]] # TODO changed for Py3
        ae_emb[i] = geo_codes[sn_model_to_int[model_name.encode('UTF-8')]]

    return ae_emb


def remove_duplicates(dataset):
    ''' Two duplicates have the same target and context.
    Returns a new dataset that has a single utterance associated with each duplicate
    and a synchronized numpy array, that contains all the duplicates.
    '''
    dataset = dataset.clone()
    geos_grouped = group_geometries(dataset.in_geo, dataset.target)
    unique_target = []
    unique_idx = []
    duplicates = []
    
    for key, val in geos_grouped.items():
        unique_target.append(key[-1])  # target-id of group
        unique_idx.append(val[0])      # location (of first instance) in dataset.in_geo
        duplicates.append(val)
        
    # Sort according to context.  # helpful for "serial" visualization of results
    s_idx = np.argsort(unique_target)
    unique_idx = np.array(unique_idx)[s_idx]
    duplicates = np.array(duplicates)[s_idx]
    
    duplicate_utters = []
    for d in duplicates:
        duplicate_utters.append(dataset.text[d])
    
    dataset.extract(unique_idx, in_place=True)
    dataset.freeze()
    return dataset, duplicate_utters


def group_geometries(geo_ids, target_indicators=None):
    ''' If the geometries associated with an utterance are the same, group them together.
    Input:
        geo_ids (N x 3): N triplets of integers.
        target_indicators: (N x 3) indicator of which was the target geometry for each of the geo_ids.
    Returns:
         A dictionary where each key maps to the rows of the ``geo_ids`` that are comprised by the same set of integers, and same ``target_indicators``, if the latter is not None.
    '''
    if target_indicators is not None and not np.all(np.unique(target_indicators) == [0,1]):
        raise ValueError('provide one-hot indicators.')
        
    n_triplets = len(geo_ids)
    if target_indicators is not None:
        if n_triplets != len(target_indicators):
            raise ValueError()

    groups = defaultdict(list)
    for i in range(n_triplets):
        g = geo_ids[i].astype(np.int)
        if target_indicators is not None:
            t = g[np.where(target_indicators[i])]            
            key = tuple(np.hstack([sorted(g), t]))
        else:
            key = tuple(sorted(g))
        groups[key].append(i)
    return groups


def group_target_geometries(geo_ids, indicators):
    ''' Returns a dictionary mapping each geo_id at the rows of geo_ids of which it was 
    used as a target. I.e., the values of each key in the result are an equivalence class.
    '''
    
    if not np.all(np.unique(indicators) == [0, 1]):
        raise ValueError('provide one-hot indicators.')
    
    groups = defaultdict(list) # Hold for each target_geometry the rows which is used.
    n_triplets = len(geo_ids)
    for i in range(n_triplets):
        target_geo_i = geo_ids[i][np.where(indicators[i])][0]    
        groups[target_geo_i].append(i)
    return groups


def far_or_close_dict(game_data):    
    geo_ids = np.array(game_data[['chair_a', 'chair_b', 'chair_c']], dtype=np.int32)
    context = np.array(game_data.context_condition)
    res = dict()
    for i in range(len(geo_ids)):
        key = tuple(sorted(geo_ids[i]))
        val = context[i] == 'close'
        if key in res:
            assert( res[key] == val )
        else:
            res[key] = val
    return res
    
    
def is_close(triplet=None, game_data=None):
    '''A triplet (geo_ids) can be either far or close.
    ''' 
    if not hasattr(is_close, 'state'):
        is_close.state = far_or_close_dict(game_data)
    
    if triplet is not None:
        return is_close.state[tuple(sorted(triplet))]
    
    
def context_based_subset_of_dataset(dataset, is_close, extract_close=True):
    ''' Split a dataset in utterances based on their context: far or close.
        far_or_close: function evaluating if a triplet is far or close.
        extract_close: if True, the subset is consisted by close only triplets.
    '''
    subset = dataset.clone()
    subset_mask = np.zeros(subset.n_examples, dtype=np.bool)
    for i in range(subset.n_examples):
        subset_mask[i] = is_close(subset.in_geo[i])
    
    if not extract_close:
        subset_mask = np.logical_not(subset_mask)        

    subset.apply_mask(subset_mask)
    return subset


def split_examples_in_target_set(in_dataset, target_set):
    ''' Splits input dataset into to two disjoint datasets: the first contains
    triplets for which the target ids are in the the input targets and the second
    their complement.
    '''    
    n_check = in_dataset.n_examples
    mask = np.zeros(n_check, dtype=np.bool)    
    for i in range(n_check):
        target_index = np.where(in_dataset.target[i])[0]
        if len(target_index) != 1:
            raise ValueError('')
        target_index = target_index[0]
        if in_dataset.in_geo[i][target_index] in target_set:
            mask[i] = True
            
    pos = in_dataset.clone()
    neg = in_dataset.clone()
    pos.apply_mask(mask)
    neg.apply_mask(np.logical_not(mask))
    return pos, neg


def shuffle_game_geometries(geo_ids, labels, parts=None, random_seed=None):
    ''' e.g. if [a, b, c] with label 1 makes it [b, a, c] with label 0.
    '''
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffle = np.random.shuffle
    for i in range(len(geo_ids)):
        idx = [0, 1, 2]
        shuffle(idx)
        geo_ids[i] = geo_ids[i][idx]
        labels[i] = labels[i][idx]
        if parts is not None:
            parts[i] = parts[i][idx]

    if parts is not None:
        return geo_ids, labels, parts
    else:
        return geo_ids, labels


def convert_labels_to_one_hot(labels, n_classes=3):
    # Convert labels to indicators.
    n_utter = len(labels)
    targets = np.array(labels, dtype=np.int32)
    target_oh = np.zeros(shape=(n_utter, n_classes))
    for i in range(n_utter):
        target_oh[i, targets[i]] = 1
    assert(np.all(np.sum(target_oh, axis=1) == 1))
    return target_oh
