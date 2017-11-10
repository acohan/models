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

"""Sequence-to-Sequence with attention model for text summarization.
"""
from collections import namedtuple

import numpy as np
import seq2seq_lib
from six.moves import xrange
import tensorflow as tf

from attention_decoder import attention_decoder
from tensorflow.python import debug as tf_debug


HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, max_grad_norm, '
                     'num_softmax_samples, trunc_norm_init_std, '
                     )


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.
  A function that feeds previous model output rather than ground truth

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  # Used in `tf.contrib.legacy_seq2seq.attention_decoder`
  # Has this signature: loop_function(prev, i) = next
  #   prev: 2D Tensor of shape (batch_size x output_size)
  #   i is an integer, the step number (when advanced control is needed)
  #   next is a 2D Tensor of shape [batch_size, input_size]
  def loop_function(prev, _):
    """function that feed previous model output rather than ground truth."""
    if output_projection is not None:
      prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = tf.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab, num_gpus=0):
    self._hps = hps
    self._vocab = vocab
    self._num_gpus = num_gpus
    self._cur_gpu = 0
    self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

  def run_train_step(self, sess, article_batch, abstract_batch, targets,
                     article_lens, abstract_lens, loss_weights):
    # run the optimizer, summary writers, get loss, global step
    to_return = [self._train_op, self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def run_eval_step(self, sess, article_batch, abstract_batch, targets,
                    article_lens, abstract_lens, loss_weights):
    # run summary writers, loss and global step
    to_return = [self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def run_decode_step(self, sess, article_batch, abstract_batch, targets,
                      article_lens, abstract_lens, loss_weights):
    to_return = [self._outputs, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def _next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self._num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self._cur_gpu
    if self._num_gpus > 1:
      self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus-1)
    return dev

  def _get_gpu(self, gpu_id):
    if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
      return ''
    return '/gpu:%d' % gpu_id

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self._hps
    self._articles = tf.placeholder(tf.int32,
                                    [hps.batch_size, hps.enc_timesteps],
                                    name='articles')
    self._abstracts = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps],
                                     name='abstracts')
    self._targets = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='targets')
    self._article_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='article_lens')
    self._abstract_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                         name='abstract_lens')
    self._loss_weights = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                        name='loss_weights')

  def _add_seq2seq(self):
    hps = self._hps
    vsize = self._vocab.NumIds()

    with tf.variable_scope('seq2seq'):
      # unstacks the articles, abstracts, targets, etc into a list of len=time_steps.
      encoder_inputs = tf.unstack(tf.transpose(self._articles))
      decoder_inputs = tf.unstack(tf.transpose(self._abstracts))
      targets = tf.unstack(tf.transpose(self._targets))
      loss_weights = tf.unstack(tf.transpose(self._loss_weights))
      article_lens = self._article_lens

      # Embedding shared by the input and outputs.
      # embedds words in the encoder and decoder
      with tf.variable_scope('embedding'), tf.device('/cpu:0'):
        embedding = tf.get_variable(
            'embedding', [vsize, hps.emb_dim], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in encoder_inputs]
        emb_encoder_inputs = tf.stack(emb_encoder_inputs, axis=1)
        emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in decoder_inputs]
      # stack n layers of lstms for encoder
      for layer_i in xrange(hps.enc_layers):
        with tf.variable_scope('encoder%d'%layer_i), tf.device(
            self._next_device()):
          cell_fw = tf.contrib.rnn.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
              state_is_tuple=True)
          cell_bw = tf.contrib.rnn.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
              state_is_tuple=True)

#           cell_fw = tf.Print(cell_fw, [cell_fw[0].shape], message='shape of cell_fw:')
#           cell_bw = tf.Print(cell_bw, [cell_bw[0].shape], message='shape of cell_bw:')
#           emb_encoder_inputs = tf.Print(emb_encoder_inputs, [len(emb_encoder_inputs), emb_encoder_inputs[0].shape], message='shape of emb_in:')
#           emb_decoder_inputs = tf.Print(emb_decoder_inputs, [len(emb_decoder_inputs), emb_decoder_inputs[0].shape], message='shape of emb_out:')
#           article_lens = tf.Print(article_lens, [article_lens.shape], message='article lens')
#           (emb_encoder_inputs, fw_state, _) = tf.contrib.rnn.static_bidirectional_rnn(
#               cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32,
#               sequence_length=article_lens)
          (emb_encoder_inputs, (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
              cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32,
              sequence_length=article_lens, swap_memory=True)
          emb_encoder_inputs = tf.concat(axis=2, values=emb_encoder_inputs) # concatenate the forwards and backwards states
      encoder_outputs = emb_encoder_inputs
      # reduce the decoder states, use an MLP to transform the fw/bw states into one
      with tf.variable_scope('reduce_final_state'):
        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable(
            'w_reduce_c', [hps.num_hidden * 2, hps.num_hidden],
            dtype=tf.float32, initializer=self.trunc_norm_init)
        w_reduce_h = tf.get_variable(
            'w_reduce_h', [hps.num_hidden * 2, hps.num_hidden],
            dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_reduce_c = tf.get_variable(
            'bias_reduce_c', [hps.num_hidden], dtype=tf.float32,
            initializer=self.trunc_norm_init)
        bias_reduce_h = tf.get_variable(
            'bias_reduce_h', [hps.num_hidden],
            dtype=tf.float32, initializer=self.trunc_norm_init)
        
        
        old_c = tf.concat(axis=1, values=[fw_state.c, bw_state.c])
        old_h = tf.concat(axis=1, values=[fw_state.h, bw_state.h])
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
        encoder_output_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      

      # define a weight matrix to project output of hidden state to vocabulary (w=[num_hiddn x vocab_size], biases=v)
      with tf.variable_scope('output_projection'):
        w = tf.get_variable(
            'w', [hps.num_hidden, vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        w_t = tf.transpose(w)
        v = tf.get_variable(
            'v', [vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

      with tf.variable_scope('decoder'), tf.device(self._next_device()):
        # When decoding, use model output from the previous step
        # for the next step. In training just use the direct inputs
        loop_function = None
        if hps.mode == 'decode':
          loop_function = _extract_argmax_and_embed(
              embedding, (w, v), update_embedding=False)

        cell = tf.contrib.rnn.LSTMCell(
            hps.num_hidden,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
            state_is_tuple=True)
        # reshape encoder_outputs
        # the outputs is a list of shapes [batch_size x 2*num_hidden]
        # 2*num_hidden because we have a bidirectional rnn and it concats the outputs
        # we want convert the list of shapes into a single tensor where the second dimension is time_steps
        # add a new dimension at second dimension : [batch_size, 1, 2*num_hidden]
#         encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, 2*hps.num_hidden])
#                            for x in encoder_outputs]
#         # then concat all the time_steps along that axis: shape=[batch_size, time_steps, 2*num_hidden]
#         self._enc_states = tf.concat(axis=1, values=encoder_outputs)
        # last step of the fw and bw rnn for decoder input
        self._enc_states = encoder_outputs
        self._dec_in_state = encoder_output_state
        # During decoding, follow up _dec_in_state are fed from beam_search.
        # dec_out_state are stored by beam_search for next step feeding.
        initial_state_attention = (hps.mode == 'decode')
        # during decoding, the RNN can look up information in the additional tensor, attention_states
        # Next decode using attention
        # decoder_outputs is a list of tensorf of shape [batch_size x output_size]
        # TODO: check actually how does the `attention_decoder` works
        decoder_outputs, self._dec_out_state = attention_decoder(
            decoder_inputs=emb_decoder_inputs,  # a list of 2D tensorfs [batch_size x embedding_size]
            initial_state=self._dec_in_state,  # 2D Tensor [batch_size, cell.state_size]
            attention_states=self._enc_states,  # 3D Tensor [batch_size, attn_length x attn_size], attn_length here is the time_steps, attn_size is the size of the rnn output (2*num_hidden)
            cell=cell,
            num_heads=1,  # number of attention heads that read from attention_states
            loop_function=loop_function,  # this function will be applied to i-th output in order to generate i+1 th input and decoder_inputs will be ignored, except for the first element (GO symbol). This can be also used in training to emulate,
            initial_state_attention=initial_state_attention)
        

      # get the output of the decoder and project it into the vocabulary matrix (output = w*output
      with tf.variable_scope('output'), tf.device(self._next_device()):
        # vocab_scores is the vocabulary distribution before applying softmax.
        # Each entry on the list corresponds to one decoder step
        model_outputs = [] 
        for i in xrange(len(decoder_outputs)):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          model_outputs.append(
              tf.nn.xw_plus_b(decoder_outputs[i], w, v))
#         vocab_dists = [tf.nn.softmax(s) for s in model_outputs]
#         log_dists = [tf.log(dist) for dist in vocab_dists]

      if hps.mode == 'decode':
        with tf.variable_scope('decode_output'):
          # get the most probable word along the vocabulary.
          self.model_outputs = model_outputs[0]
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
          self._outputs = tf.concat(
              axis=1, values=[tf.reshape(x, [hps.batch_size, 1]) for x in best_outputs])
          
        
          
          # ------------------------------------  
#           model_outputs[0] = tf.Print(model_outputs[0], [len(model_outputs), model_outputs[0].shape, model_outputs], summarize=20, message='modeloutputs:')
#           model_outputs[-1] = tf.Print(model_outputs[-1], [model_outputs[-1]], summarize=20, message="Model output: ")
          self._topk_log_probs, self._topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), hps.batch_size*2)

      # define loss, using sampled loss instead of full softmax
      with tf.variable_scope('loss'), tf.device(self._next_device()):
        def sampled_loss_func(inputs, labels):
          with tf.device('/cpu:0'):  # Try gpu.
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                weights=w_t, biases=v, labels=labels, inputs=inputs,
                num_sampled=hps.num_softmax_samples, num_classes=vsize)

        if hps.num_softmax_samples != 0 and hps.mode == 'train':
          self._loss = seq2seq_lib.sampled_sequence_loss(
              decoder_outputs, targets, loss_weights, sampled_loss_func)
        else:
          self._loss = tf.contrib.legacy_seq2seq.sequence_loss(
              model_outputs, targets, loss_weights)
        tf.summary.scalar('loss', tf.minimum(12.0, self._loss))

  def _add_train_op(self):
    """Sets self._train_op, op to run for training."""
    # set learning rate get gradients, add optimzer
    hps = self._hps

    self._lr_rate = tf.maximum(
        hps.min_lr,  # min_lr_rate.
        tf.train.exponential_decay(hps.lr, self.global_step, 30000, 0.98))

    # Returns all variables created with trainable=True.=
    tvars = tf.trainable_variables()
    with tf.device(self._get_gpu(self._num_gpus-1)):
      # tf.gradients: Constructs symbolic partial derivatives of 
      # sum of `ys` w.r.t. x in `xs`.
      grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self._loss, tvars), hps.max_grad_norm)
    tf.summary.scalar('global_norm', global_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
    tf.summary.scalar('learning rate', self._lr_rate)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')

  def run_encoder(self, sess, enc_inputs, enc_len):
    """Return the top states from encoder for decoder.

    Args:
      sess: tensorflow session.
      enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
      enc_len: encoder input length of shape [batch_size]
    Returns:
      enc_top_states: The top level encoder states.
      dec_in_state: The decoder layer initial state.
    """
    results = sess.run([self._enc_states, self._dec_in_state],
                       feed_dict={self._articles: enc_inputs,
                                  self._article_lens: enc_len})
    enc_states = results[0]
    dec_in_state = results[1]
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state
#     enc_states = results[0]
#     dec_in_state = results[1]
#     def _debug_func(enc_states, dec_in_state_0, dec_in_state_1):
#         import pdb; pdb.set_trace()  # XXX BREAKPOINT
#         from IPython import embed; embed()  # XXX DEBUG
#         return False
#     debug_op = tf.py_func(_debug_func, [enc_states, dec_in_state[0], dec_in_state[1]], [tf.bool])
#     with tf.control_dependencies(debug_op):
#         enc_states = tf.identity(enc_states, name='enc_states')
#     return enc_states, results[1]

  def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
    """Return the topK results and new decoder states."""

    
#     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
#     test_state = dec_init_states[0]
# #     test_state = tf.Print(test_state, [test_state.c, test_state.h], summarize=20, message='Decoder init state:')
#     
#         # ---------- debug -------------------
#     def _debug_func(test_state):
#       import pdb; pdb.set_trace()  # XXX BREAKPOINT
#       from IPython import embed; embed()  # XXX DEBUG
#       return False
#      
#     # Set up the debugger
#     debug_op = tf.py_func(_debug_func, [test_state], [tf.bool])
#     # Hook the debugger into the computational graph to use it at running time
#     with tf.control_dependencies(debug_op):
#       test_state = tf.identity(test_state)
    
    # ------------------------------------  
    
    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
#     cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
#     hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]

    # dec_init_states is a list of LSTMStateTuple with shape [batch x hidden], len of list = beam
#     cells = [state.c for state in dec_init_states]
#     hiddens = [state.h for state in dec_init_states]
#     new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
#     new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
#     new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h) 
#     print('shape of new_c',new_c.shape)
    
    
        # ---------- debug -------------------
#     def _debug_func(model_outputs):
#       import pdb; pdb.set_trace()  # XXX BREAKPOINT
#     #   from IPython import embed; embed()  # XXX DEBUG
#       return False
#      
#     # Set up the debugger
#     debug_op = tf.py_func(_debug_func, [self.model_outputs[0]], [tf.bool])
#     # Hook the debugger into the computational graph to use it at running time
#     with tf.control_dependencies(debug_op):
#       self.model_outputs[0] = tf.identity(self.model_outputs[0], name='model_outputs')
    
#     def _debug_func(model_outputs):
#       import pdb; pdb.set_trace()  # XXX BREAKPOINT
#       from IPython import embed; embed()  # XXX DEBUG
#       return False
#     self.model_outputs = tf.py_func(_debug_func, [self.model_outputs], [tf.bool])
#     feed = {
#         self._enc_states: enc_top_states,
#         self._dec_in_state:
#             np.squeeze(np.array(dec_init_states)),
#         self._abstracts:
#             np.transpose(np.array([latest_tokens])),
#         self._abstract_lens: np.ones([len(dec_init_states)], np.int32)}
#     states_c = np.array([s.c for s in dec_init_states])
#     states_h = np.array([s.h for s in dec_init_states])
    beam_size = len(dec_init_states)

    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
    
    feed = {
        self._enc_states: enc_top_states,
        self._dec_in_state:new_dec_in_state,
        self._abstracts:
            np.transpose(np.array([latest_tokens])),
        self._abstract_lens: np.ones([len(dec_init_states)], np.int32)}
#     self.model_outputs = tf.Print(self.model_outputs, [self.model_outputs], message='Model outputs:')
#     self.model_outputs.c = tf.Print(self.model_outputs.c, [self.model_outputs.c.shape], message='Model outputs c:')
#     self.model_outputs.h = tf.Print(self.model_outputs.h, [self.model_outputs.h.shape], message='Model outputs h:')
#     sess.run(self.model_outputs,feed_dict=feed)
#     sess.run([new_dec_in_state],feed_dict=feed)
    results = sess.run(
        [self._topk_ids, self._topk_log_probs, self._dec_out_state],
        feed_dict=feed)

    ids, probs, states = results[0], results[1], results[2]
    new_states = [
      tf.contrib.rnn.LSTMStateTuple(states.c[i, :], states.h[i, :])
      for i in xrange(beam_size)
    ]
    
    return ids, probs, new_states

  def build_graph(self):
    self._add_placeholders()
    self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
