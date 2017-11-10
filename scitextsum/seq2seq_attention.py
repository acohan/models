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
import os

"""Trains a seq2seq model.

WORK IN PROGRESS.

Implement "Abstractive Text Summarization using Sequence-to-sequence RNNS and
Beyond."

"""
import sys
import time

import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_id_key', 'article_id',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('article_key', 'article_body',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'abstract',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('labels_key', 'labels',
                           'tf.Example feature key for labels.')
tf.app.flags.DEFINE_string('section_names_key', 'section_names',
                           'tf.Example feature key for section names.')
tf.app.flags.DEFINE_string('sections_key', 'sections',
                           'tf.Example feature key for sections.')

tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 150,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', True,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')


def _running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses.
  via exponential decay.
  This is used to implement early stopping w.r.t. a more
  smooth loss curve than the raw loss curve."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
  return running_avg_loss


def _Train(model, data_batcher):
  """Runs model training."""
  with tf.device('/gpu:0'):
    model.build_graph()
    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    train_log_dir = os.path.join(FLAGS.log_root, 'traindir')
    summary_writer = tf.summary.FileWriter(train_log_dir)
    sv = tf.train.Supervisor(logdir=train_log_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    tf.logging.info("Created session.")
    running_avg_loss = 0
    step = 0
    # Training loop
    try:
      while not sv.should_stop() and step < FLAGS.max_run_steps:
        # get the next batch
        (article_batch, abstract_batch, targets, article_lens, abstract_lens,
         loss_weights, _, _, article_id) = data_batcher.NextBatch()
        # training step, runs the optimizer.apply_gradients
        t0 = time.time()
        (_, summaries, loss, train_step) = model.run_train_step(
            sess, article_batch, abstract_batch, targets, article_lens,
            abstract_lens, loss_weights)
        t1 = time.time()
        tf.logging.info('seconfs for training step: {:.3f}'.format(t1-t0))
        
        summary_writer.add_summary(summaries, train_step)
        running_avg_loss = _running_avg_loss(
            running_avg_loss, loss, summary_writer, train_step)
        step += 1
        if step % 100 == 0:
          summary_writer.flush()
          print('training loss: {:.3f}, avg loss: {:.3f}'.format(loss, running_avg_loss))
    except KeyboardInterrupt:
      tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
      sv.stop()
    sv.Stop()
    return running_avg_loss


def _Eval(model, data_batcher, vocab=None):
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  
  # allow soft placement on CPU if no gpu is available
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  best_loss = None

  eval_log_dir = os.path.join(FLAGS.log_root, "evaldir")
  summary_writer = tf.summary.FileWriter(eval_log_dir)
  bestmodel_save_path = os.path.join(eval_log_dir, 'bestmodel')
  
  while True:
    time.sleep(FLAGS.eval_interval_secs)
    try:
      train_log_dir = os.path.join(FLAGS.log_root, 'traindir')
      ckpt_state = tf.train.get_checkpoint_state(train_log_dir)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    (article_batch, abstract_batch, targets, article_lens, abstract_lens,
     loss_weights, _, _) = data_batcher.NextBatch()
    t0=time.time()

    (summaries, loss, train_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    
    t1=time.time()
    tf.logging.info('seconds for batch: {:.2f}'.format(t1-t0))

    tf.logging.info('loss: {:.3f}'.format(loss))
    tf.logging.info(
        'article:  %s',
        ' '.join(data.ids2words(article_batch[0][:].tolist(), vocab)))
    tf.logging.info(
        'abstract: %s',
        ' '.join(data.ids2words(abstract_batch[0][:].tolist(), vocab)))

    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _running_avg_loss(
        running_avg_loss, loss, summary_writer, train_step)
    
    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    if step % 100 == 0:
      summary_writer.flush()


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  vocab = data.Vocab(FLAGS.vocab_path, 1000000)
  # Check for presence of required special tokens.
  assert vocab.CheckVocab(data.PAD_TOKEN) > 0
  assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
  assert vocab.CheckVocab(data.START_DECODING) > 0
  assert vocab.CheckVocab(data.STOP_DECODING) > 0

  batch_size = 4
  if FLAGS.mode == 'decode':
    batch_size = FLAGS.beam_size

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=batch_size,
      enc_layers=1,
      enc_timesteps=800,
      dec_timesteps=200,
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=256,  # for rnn cell
      emb_dim=128,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=4096, # If 0, no sampled softmax.
      trunc_norm_init_std=0.05) 

  batcher = batch_reader.Batcher(
      FLAGS.data_path, vocab, hps,
      FLAGS.article_id_key,
      FLAGS.article_key,
      FLAGS.abstract_key,
      FLAGS.labels_key,
      FLAGS.section_names_key,
      FLAGS.sections_key,
      FLAGS.max_article_sentences,
      FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input)
  tf.set_random_seed(FLAGS.random_seed)

  if hps.mode == 'train':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Train(model, batcher)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Eval(model, batcher, vocab=vocab)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    decoder = seq2seq_attention_decode.BeamSearchDecoder(model, batcher, hps, vocab)
    decoder.decode_loop()


if __name__ == '__main__':
  tf.app.run()
