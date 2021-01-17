"""
The MIT License (MIT)
Originally created sometime in late 2018.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import os.path as osp
import numpy as np
from PIL import Image
from shapeglot.language.helper import token_ints_to_sentence
from shapeglot.in_out.helper import snc_category_to_synth_id

####
top_image_dir = 'set_to_folder_containing_images_of_shapes'
####

def stack_images_horizontally(file_names, save_file=None):
    """Opens the images corresponding to file_names and
    creates a new image stacking them horizontally.
    """
    images = list(map(Image.open, file_names))
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_file is not None:
        new_im.save(save_file)
    return new_im


def visualize_shapes_generator(data_to_vis, model_ids, top_image_dir, image_view_tag):
    for i in data_to_vis:
        image_files = []
        try:
            for j in i:
                image_files.append(osp.join(top_image_dir, model_ids[j], image_view_tag))
        except:
            image_files.append(osp.join(top_image_dir, model_ids[i], image_view_tag))
        yield stack_images_horizontally(image_files)


def visualize_triplet(triplet, sorted_model_ids, class_name='chair', image_view_tag='image_p020_t337_r005.png'):
    syn_id = snc_category_to_synth_id()[class_name]
    im_dir = osp.join(top_image_dir, syn_id)
    return visualize_shapes_generator(triplet, sorted_model_ids, im_dir, image_view_tag)


def visualize_example(geo_ids, utterance, target, sorted_sn_models, int_to_word,
                      context=None, guess=None, image_view_tag='image_p020_t337_r005.png'):
    if utterance is not None and len(utterance) > 1:
        utterance = np.squeeze(utterance)

    if utterance is not None:
        text = token_ints_to_sentence(utterance, int_to_word)
        print('Utterance:', text)

    if context is not None:
        print('Context:', context)

    if guess is not None:
        print('Guessed correct:', guess)
    
    if target == 0:
        target_str = 'left-most'
    elif target == 1:
        target_str = 'middle'
    else:
        target_str = 'right-most'
    
    print('Target:', target_str)
    return next(visualize_triplet([geo_ids], sorted_sn_models, image_view_tag=image_view_tag))


def visualize_game_example(game_data, utterance_id, sorted_sn_models, int_to_word):
    text = game_data.text[utterance_id]
    target = game_data.target_chair[utterance_id]
    context = game_data.context_condition[utterance_id]
    guess = game_data.correct[utterance_id]

    a = game_data.chair_a[utterance_id]
    b = game_data.chair_b[utterance_id]
    c = game_data.chair_c[utterance_id]
    geo_ids = [a, b, c]
    return visualize_example(geo_ids, text, target, sorted_sn_models, int_to_word, context=context, guess=guess)