'''
Created on August 29, 2017

@author: optas
'''

import numpy as np
import string
import copy
from six.moves import xrange

from shapeglot.simple_utils import iterate_in_chunks


def _all_tensors_have_same_rows(tensor_list):
    n = tensor_list[0].shape[0]
    for i in xrange(1, len(tensor_list)):
        if n != tensor_list[i].shape[0]:
            return False
    return True

num2alpha = dict(enumerate(string.ascii_lowercase, 0))


class NumpyDataset(object):
    """ Handy class for working with data that can fit in memory and which can be represented
    via numpy tensors. It was created to play well with Tensorflow before the introduciton
    of the Dataset class (around 2018).
    """
    def __init__(self, tensor_list, tensor_names=None, copy=True, init_shuffle=True):
        if tensor_names is not None and len(tensor_names) != len(tensor_list):
            raise ValueError('Each tensor must have a name or none of them has.')

        if not _all_tensors_have_same_rows(tensor_list):
            raise ValueError('Tensors must have the number of elements')

        if len(tensor_list) > len(num2alpha):
            raise ValueError('Too many non-named tensors.')

        self.n_tensors = len(tensor_list)
        self.n_examples = tensor_list[0].shape[0]

        if tensor_names is None:
            tensor_names = [num2alpha[i] for i in range(self.n_tensors)]

        self.tensor_names = tensor_names

        for name, val in zip(self.tensor_names, tensor_list):
            if copy:
                self.__setattr__(name, val.copy())
            else:
                self.__setattr__(name, val)

        self.epochs_completed = 0
        self._index_in_epoch = 0
        self._frozen = False
        if init_shuffle:
            self.shuffle_data()

    def __str__(self):
        res = ''
        for name in self.tensor_names:
            res += name + ' ' + str(self.__getattribute__(name).shape) + '\n'
        return res
    
    def __len__(self):
        return self.n_examples

    def __getitem__(self, key):
        return getattr(self, key)
                
    def shuffle_data(self, seed=None):
        if self._frozen:
            return self

        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.n_examples)
        np.random.shuffle(perm)

        for name in self.tensor_names:
            self.__setattr__(name, self.__getattribute__(name)[perm])
        return self

    def add_tensor(self, new_tensor, tensor_name):
        a_tensor = self.__getattribute__(self.tensor_names[0])
        if a_tensor.shape[0] != new_tensor.shape[0]:
            raise ValueError('Input tensor has different number of rows than established tensors.')

        if tensor_name in self.tensor_names:
            raise ValueError('Tensor with the same name already exists.')

        self.__setattr__(tensor_name, new_tensor)
        self.tensor_names.append(tensor_name)
        return self
    
    def delete_tensor(self, tensor_name):
        if tensor_name not in self.tensor_names:
            raise ValueError('Tensor with this name doesn\'t exists.')
        delattr(self, tensor_name)        
        self.tensor_names.remove(tensor_name)
        return self

    def next_batch(self, batch_size, tensor_names=None, seed=None):
        '''Return the next batch_size examples from this data set.

        tensor_names: (list of strings, default=None) describes the names of the tensors that will be returned.
        If none, all tensors will be batched.
        '''
        assert batch_size <= self.n_examples
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size            
        end = self._index_in_epoch

        ret_res = []

        if tensor_names is not None:
            tensors_to_look = tensor_names
        else:
            tensors_to_look = self.tensor_names     # I.e. use all of them.

        for name in tensors_to_look:
            ret_res.append(self.__getattribute__(name)[start:end])

        if self.n_tensors == 1:
            ret_res = ret_res[0]

        return ret_res

    def freeze(self):
        if self._frozen:
            raise ValueError('Dataset is already frozen.')
        self._frozen = True
        return self

    def unfreeze(self):
        if not self._frozen:
            raise ValueError('Dataset is not frozen.')
        self._frozen = False
        return self

    def is_equal(self, other_dataset):

        if other_dataset.n_examples != self.n_examples or \
           other_dataset.n_tensors != self.n_tensors or \
           np.all(other_dataset.tensor_names != self.tensor_names):
            return False

        for name in self.tensor_names:
            if not np.all(np.sort(self.__getattribute__(name), axis=0) ==\
                          np.sort(other_dataset.__getattribute__(name), axis=0)):
                return False
        return True
    
    def epoch_iterator(self, batch_size, tensor_names=None):
        ''' Returns an iterator all tensors, or those specified in tensor_names, 
        in chunks of batch_size (with the exception of the last one which can be smaller.)
        doesn't use/effect any randomization.
        '''
        if tensor_names is not None:
            tensors_to_look = tensor_names
        else:
            tensors_to_look = self.tensor_names     # I.e. use all of them.
        
        n_tensors = len(tensors_to_look)
    
        for chunk in iterate_in_chunks(np.arange(self.n_examples), batch_size):
            ret_res = []
            
            for name in tensors_to_look:
                ret_res.append(self.__getattribute__(name)[chunk])
            
            if n_tensors == 1:
                yield ret_res[0]
            else:
                yield ret_res
        
    
    def all_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire 
        data set (i.e. an epoch's data), shuffled.
        DEPRECATED.
        '''

        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.n_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)

        ret_res = []
        for name in self.tensor_names:
            ret_res.append(self.__getattribute__(name)[perm])
        return ret_res

    def merge(self, other_data_set):
        self._index_in_epoch = 0
        self.epochs_completed = 0

        for name in self.tensor_names:
            p_self = self.__getattribute__(name)
            p_other = other_data_set.__getattribute__(name)
            
            if p_self.ndim == 1:
                merger = np.hstack
            else:
                merger = np.vstack
            
            merged_prop = merger([p_self, p_other])
            self.__setattr__(name, merged_prop)

        self.n_examples = self.n_examples + other_data_set.n_examples

        return self

    def clone(self):
        tensor_list = self.all_data(shuffle=False)
        tensor_names = copy.deepcopy(self.tensor_names)
        return NumpyDataset(tensor_list, tensor_names, init_shuffle=False, copy=True)

    def subsample(self, n_samples, replace=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        new_dataset = self.clone()
        all_ids = np.arange(new_dataset.n_examples)
        sub_ids = np.random.choice(all_ids, n_samples, replace=replace)

        for name in new_dataset.tensor_names:
            sub_prop = new_dataset.__getattribute__(name)[sub_ids]
            new_dataset.__setattr__(name, sub_prop)

        new_dataset.n_examples = n_samples
        return new_dataset
    
    def extract(self, indices, in_place=False):
        ''' Constrain the tensors along the given indices (on zero-th dimension).
        '''        
        if np.max(indices) >= self.n_examples or np.min(indices) < 0:
            raise ValueError('Not appropriate indices provided.')
                    
        if in_place:
            new_dataset = self
        else:
            new_dataset = self.clone()
        
        for name in new_dataset.tensor_names:
            sub_prop = new_dataset.__getattribute__(name)[indices]
            new_dataset.__setattr__(name, sub_prop)
        
        new_dataset.n_examples = len(indices)
        return new_dataset
    
    def apply_mask(self, mask):
        ''' boolean mask of self.n_examples will be applied to its stored tensor.
        '''
        for name in self.tensor_names:
            self.__setattr__(name, self.__getattribute__(name)[mask])
        self.n_examples = int(np.sum(mask))
        assert(self.n_examples == len(self.__getattribute__(name)))
        return self
    
    def permute_tensors(self, permutation):
        if np.any(sorted(permutation) != np.arange(self.n_examples)):
            raise ValueError('Not proper permutation was given.')

        for name in self.tensor_names:
            self.__setattr__(name, self.__getattribute__(name)[permutation])
