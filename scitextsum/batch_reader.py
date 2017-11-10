# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Batch reader to seq2seq attention model, with bucketing support."""

from collections import namedtuple
from random import shuffle
from threading import Thread
import time

import numpy as np
import six
from six.moves import queue as Queue
from six.moves import xrange
import tensorflow as tf

import data

# ModelInput = namedtuple('ModelInput',
#                         'enc_input dec_input target enc_len dec_len '
#                         'origin_article origin_abstract')

class ModelInput(object):
  
  def __init__(self,
               enc_input, dec_input, target, enc_len, dec_len,
               original_article, original_abstract, article_id):
    self.enc_input = enc_input
    self.dec_input = dec_input
    self.target = target
    self.enc_len = enc_len
    self.dec_len = dec_len
    self.original_article = original_article
    self.original_abstract = original_abstract
    self.article_id = article_id

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100

# To represent list of sections as string and retrieve it back
SECTION_SEPARATOR = ' <SCTN/> '

# to represent separator as string, end of item (ei)
LIST_SEPARATOR = ' <EI/> '


def _string_to_list(s, dtype='str'):
  """ converts string to list
  Args:
    s: input
    dtype: specifies the type of elements in the list
      can be one of `str` or `int`
  """
  if dtype == 'str':
    return s.split(LIST_SEPARATOR)
  elif dtype == 'int':
    return [int(e) for e in s.split(LIST_SEPARATOR) if e]


def _string_to_nested_list(s):
  return [e.split(LIST_SEPARATOR)
          for e in s.split(SECTION_SEPARATOR)]

class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self, data_path, vocab, hps,
               article_id_key,
               article_key, abstract_key,
               labels_key,
               section_names_key,
               sections_key,
               max_article_sentences,
               max_abstract_sentences, bucketing=True, truncate_input=False):
    """Batcher constructor.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary.
      hps: Seq2SeqAttention model hyperparameters.
      article_id_key: article id key in tf.Example
      article_key: article feature key in tf.Example.
      abstract_key: abstract feature key in tf.Example.
      labels_key: labels feature key in tf.Example,
      section_names_key: section names key in tf.Example,
      sections_key: sections key in tf.Example,
      max_article_sentences: Max number of sentences used from article.
      max_abstract_sentences: Max number of sentences used from abstract.
      bucketing: Whether bucket articles of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._article_id_key = article_id_key
    self._article_key = article_key
    self._abstract_key = abstract_key
    self._labels_key = labels_key
    self._section_names_key = section_names_key
    self._sections_key = sections_key
    self._max_article_sentences = max_article_sentences
    self._max_abstract_sentences = max_abstract_sentences
    self._bucketing = bucketing
    self._truncate_input = truncate_input
    self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
    self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)
    self._input_threads = []
    for _ in xrange(16):
      self._input_threads.append(Thread(target=self._fill_input_queue))
      self._input_threads[-1].daemon = True
      self._input_threads[-1].start()
    self._bucketing_threads = []
    for _ in xrange(4):
      self._bucketing_threads.append(Thread(target=self._fill_bucket_input_queue))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()

    self._watch_thread = Thread(target=self._watch_threads)
    self._watch_thread.daemon = True
    self._watch_thread.start()

  def NextBatch(self):
    """Returns a batch of inputs for seq2seq attention model.

    Returns:
      enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
      dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].
      target_batch: A batch of targets [batch_size, hps.dec_timestamps].
      enc_input_len: encoder input lengths of the batch.
      dec_input_len: decoder input lengths of the batch.
      loss_weights: weights for loss function, 1 if not padded, 0 if padded.
      origin_articles: original article words.
      origin_abstracts: original abstract words.
    """
    enc_batch = np.zeros(
        (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
    enc_input_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    dec_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    dec_output_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    target_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    loss_weights = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.float32)
    origin_articles = ['None'] * self._hps.batch_size
    origin_abstracts = ['None'] * self._hps.batch_size
    article_ids = ['None'] * self._hps.batch_size

    buckets = self._bucket_input_queue.get()
    for i in xrange(self._hps.batch_size):
      enc_inputs= buckets[i].enc_input
      dec_inputs = buckets[i].dec_input
      targets = buckets[i].target
      enc_input_len = buckets[i].enc_len
      dec_output_len = buckets[i].dec_len
      article = buckets[i].original_article
      abstract = buckets[i].original_abstract


      origin_articles[i] = article
      origin_abstracts[i] = abstract
      enc_input_lens[i] = enc_input_len
      dec_output_lens[i] = dec_output_len
      enc_batch[i, :] = enc_inputs[:]
      dec_batch[i, :] = dec_inputs[:]
      target_batch[i, :] = targets[:]
      for j in xrange(dec_output_len):
        loss_weights[i][j] = 1
      article_ids[i] = buckets[i].article_id
    return (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
            loss_weights, origin_articles, origin_abstracts, article_ids)

  def _fill_input_queue(self):
    """Fill input queue with ModelInput.
    Reads data from file and processess into tf.Examples which are then
    placed into the input queue"""
    
    start_id = self._vocab.WordToId(data.SENTENCE_START)
    end_id = self._vocab.WordToId(data.SENTENCE_END)
    pad_id = self._vocab.WordToId(data.PAD_TOKEN)
    input_gen = self._data_generator(data.ExampleGen(self._data_path))

    while True:
      (article_id, article, abstract, labels, section_names, sections) =\
        six.next(input_gen)
        
      
#       article_sentences = [sent.strip() for sent in
#                            data.ToSentences(article, include_token=False)]
#       abstract_sentences = [sent.strip() for sent in
#                             data.ToSentences(abstract, include_token=False)]

      article_sentences = article
      abstract_sentences = abstract
      enc_inputs = []
      # Use the <s> as the <GO> symbol for decoder inputs.
      dec_inputs = [start_id]
      
      # Convert first N sentences to word IDs, stripping existing <s> and </s>.
      for i in xrange(min(self._max_article_sentences,
                          len(article_sentences))):
        enc_inputs += data.get_word_ids(article_sentences[i], self._vocab)
      for i in xrange(min(self._max_abstract_sentences,
                          len(abstract_sentences))):
        dec_inputs += data.get_word_ids(abstract_sentences[i], self._vocab)

      # Filter out too-short input
      if (len(enc_inputs) < self._hps.min_input_len or
          len(dec_inputs) < self._hps.min_input_len):
        tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d',
                           len(enc_inputs), len(dec_inputs))
        continue

      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        if (len(enc_inputs) > self._hps.enc_timesteps or
            len(dec_inputs) > self._hps.dec_timesteps):
          tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d',
                             len(enc_inputs), len(dec_inputs))
          continue
      # If we are truncating input, do so if necessary
      else:
        if len(enc_inputs) > self._hps.enc_timesteps:
          enc_inputs = enc_inputs[:self._hps.enc_timesteps]
        if len(dec_inputs) > self._hps.dec_timesteps:
          dec_inputs = dec_inputs[:self._hps.dec_timesteps]

      # targets is dec_inputs without <s> at beginning, plus </s> at end
      targets = dec_inputs[1:]
      targets.append(end_id)

      # Now len(enc_inputs) should be <= enc_timesteps, and
      # len(targets) = len(dec_inputs) should be <= dec_timesteps

      enc_input_len = len(enc_inputs)
      dec_output_len = len(targets)

      # Pad if necessary
      while len(enc_inputs) < self._hps.enc_timesteps:
        enc_inputs.append(pad_id)
      while len(dec_inputs) < self._hps.dec_timesteps:
        dec_inputs.append(end_id)
      while len(targets) < self._hps.dec_timesteps:
        targets.append(end_id)

      element = ModelInput(enc_inputs, dec_inputs, targets, enc_input_len,
                           dec_output_len, ' '.join(article_sentences),
                           ' '.join(abstract_sentences), article_id)
      self._input_queue.put(element)

  def _fill_bucket_input_queue(self):
    """Fill bucketed batches into the bucket_input_queue.
    Takes Examples out of example queue, sorts them by encoder sequence length
    processes into Batches and places them in the batch queue."""
    while True:
      inputs = []
      for _ in xrange(self._hps.batch_size * BUCKET_CACHE_BATCH):
        inputs.append(self._input_queue.get())
      if self._bucketing:
        inputs = sorted(inputs, key=lambda inp: inp.enc_len)

      batches = []
      for i in xrange(0, len(inputs), self._hps.batch_size):
        batches.append(inputs[i:i+self._hps.batch_size])
      shuffle(batches)
      for b in batches: # each b is a list of Example objects
        self._bucket_input_queue.put(b)

  def _watch_threads(self):
    """Watch the daemon input threads and restart if dead."""
    while True:
      time.sleep(60)
      input_threads = []
      for t in self._input_threads:
        if t.is_alive():
          input_threads.append(t)
        else:
          tf.logging.error('Found input thread dead.')
          new_t = Thread(target=self._fill_input_queue)
          input_threads.append(new_t)
          input_threads[-1].daemon = True
          input_threads[-1].start()
      self._input_threads = input_threads

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._fill_bucket_input_queue)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()
      self._bucketing_threads = bucketing_threads

  def _data_generator(self, example_gen):
    """Generates article and abstract text from tf.Example."""
    while True:
      e = six.next(example_gen)
      try:
        article_id = self._get_example_feature(e, self._article_id_key)
        article_text = self._get_example_feature(e, self._article_key)
        abstract_text = self._get_example_feature(e, self._abstract_key)
        labels = self._get_example_feature(e, self._labels_key)
        section_names = self._get_example_feature(e, self._section_names_key)
        sections = self._get_example_feature(e, self._sections_key)
        
        # convert to list
        article_text = _string_to_list(article_text)
        abstract_text = _string_to_list(abstract_text)
        labels = _string_to_list(labels, dtype='int')
        section_names = _string_to_list(section_names) 
        sections = _string_to_nested_list(sections) # list of lists
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue

      yield (article_id, article_text, abstract_text, labels, section_names, sections)

  def _get_example_feature(self, ex, key):
    """Extract text for a feature from td.Example.

    Args:
      ex: tf.Example.
      key: key of the feature to be extracted.
    Returns:
      feature: a feature text extracted.
    """
    return ex.features.feature[key].bytes_list.value[0].decode(
            'utf-8', 'ignore')
