# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (c) 2020 Sawan Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#""" GLUE processors and helpers """
#

import logging
import os
import numpy as np

from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.file_utils import is_tf_available
from typing import List, Optional, Union
from dataclasses import dataclass

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      label_list=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True,
                                      token_label=False):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    label_map = {label: i for i, label in enumerate(label_list)}
    pad_token_label_id = -100
    label_map["<pad>"] = pad_token_label_id

    features = []
    if examples[0].text_b is not None:
        k = len(examples[0].text_b)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)

        text_a = example.text_a
        text_b = example.text_b

        
        def get_indices(a, b, l=None):
            
            if l is None:
                inputs = tokenizer.encode_plus(
                    a,
                    b,
                    add_special_tokens=True,
                    max_length=max_length,
                )
                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            else:
                pad_token_label_id = -100
                words_a = a.split(' ')
                if b is not None:
                    words_b = b.split(' ')
                    l1, l2 = l.split('\t')[0], l.split('\t')[1]
                    words_l1, words_l2 = l1.split(' '), l2.split(' ')
                else:
                    l1 = l
                    words_l1 = l1.split(' ')                
                tokens = []
                label_ids = []
                for word, label in zip(words_a, words_l1):
                    word_tokens = tokenizer.tokenize(word)
                    if len(word_tokens) > 0:
                        tokens.extend(word_tokens)
                        label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                
                
                if b is not None:
                    tokens += [tokenizer.sep_token]
                    label_ids += [pad_token_label_id]                
                    for word, label in zip(words_b, words_l2):
                        word_tokens = tokenizer.tokenize(word)
                        if len(word_tokens) > 0:
                            tokens.extend(word_tokens)
                            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

                special_tokens_count = 2
                if len(tokens) > max_length - special_tokens_count:
                    tokens = tokens[: (max_length - special_tokens_count)]
                    label_ids = label_ids[: (max_length - special_tokens_count)]
                    
                tokens += [tokenizer.sep_token]
                label_ids += [pad_token_label_id]
                
                tokens = [tokenizer.cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                
                #Roberta does not use token_type_ids
                token_type_ids = [pad_token_segment_id] * len(tokens)               
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
           

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                if l is not None:
                    label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                if l is not None:
                    label_ids = label_ids + ([pad_token_label_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
            
            if l is not None:
                assert len(label_ids) == max_length, "Error with input length {} vs {}".format(len(label_ids), max_length)
                return input_ids, attention_mask, token_type_ids, label_ids, tokens
            else:
                return input_ids, attention_mask, token_type_ids
        
        if token_label:
            input_ids, attention_mask, token_type_ids, label, tokens = get_indices(text_a, text_b, example.label)
        else:
            input_ids, attention_mask, token_type_ids = get_indices(text_a, text_b)
            label = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if token_label:
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("label_ids: %s" % " ".join([str(x) for x in label]))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label))
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

        
    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features
