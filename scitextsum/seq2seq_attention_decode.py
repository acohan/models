import os
import time

import beam_search
import data
from six.moves import xrange
import tensorflow as tf
from collections import OrderedDict
import json

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decode_steps', 1000000,
                            'Number of decoding steps.')
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 8000,
                            'Number of batches to decode before restoring next '
                            'checkpoint')

DECODE_LOOP_DELAY_SECS = 60
DECODE_IO_FLUSH_INTERVAL = 100


class DecodeIO(object):
  
  def __init__(self, outdir):
    self._outdir = outdir
    if not os.path.exists(self._outdir):
      os.mkdir(self._outdir)
  
  def write(self, ref, decoded, article, article_id):
    obj = OrderedDict({'ref': ref, 'decode': decoded, 'article': article, 'article_id':article_id})
    with open(os.path.join(self._outdir, 'dec.json'), 'a') as mf:
      mf.write(json.dumps(obj))
      mf.write('\n')

class DecodeIO_old(object):
  """Writes the decoded and references to RKV files for Rouge score.

    See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
  """

  def __init__(self, outdir):
    self._cnt = 0
    self._outdir = outdir
    if not os.path.exists(self._outdir):
      os.mkdir(self._outdir)
    self._ref_file = None
    self._decode_file = None

  def Write(self, reference, decode):
    """Writes the reference and decoded outputs to RKV files.

    Args:
      reference: The human (correct) result.
      decode: The machine-generated result
    """
    self._ref_file.write('output=%s\n' % reference)
    self._decode_file.write('output=%s\n' % decode)
    self._cnt += 1
    if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
      self._ref_file.flush()
      self._decode_file.flush()

  def ResetFiles(self):
    """Resets the output files. Must be called once before Write()."""
    if self._ref_file: self._ref_file.close()
    if self._decode_file: self._decode_file.close()
    timestamp = int(time.time())
    self._ref_file = open(
        os.path.join(self._outdir, 'ref%d'%timestamp), 'w')
    self._decode_file = open(
        os.path.join(self._outdir, 'decode%d'%timestamp), 'w')


class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batch_reader, hps, vocab):
    """Beam search decoding.

    Args:
      model: The seq2seq attentional model.
      batch_reader: The batch data reader.
      hps: Hyperparamters.
      vocab: Vocabulary
    """
    self._model = model
    self._model.build_graph()
    self._batch_reader = batch_reader
    self._hps = hps
    self._vocab = vocab
    self._saver = tf.train.Saver()
    self._decode_io = DecodeIO(FLAGS.decode_dir)

  def decode_loop(self):
    """Decoding loop for long running process."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    step = 0
    while step < FLAGS.max_decode_steps:
      time.sleep(DECODE_LOOP_DELAY_SECS)
      if not self._Decode(self._saver, sess):
        continue
      step += 1
      print('steps: {:d}/{:d}'.format(step, FLAGS.max_decode_steps), end='\r', flush=True)

  def _Decode(self, saver, sess):
    """Restore a checkpoint and decode it.

    Args:
      saver: Tensorflow checkpoint saver.
      sess: Tensorflow session.
    Returns:
      If success, returns true, otherwise, false.
    """
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
      return False

    tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(
        FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
    tf.logging.info('renamed checkpoint path %s', ckpt_path)
    saver.restore(sess, ckpt_path)

#     self._decode_io.ResetFiles()
    for _ in xrange(FLAGS.decode_batches_per_ckpt):
      (article_batch, _, _, article_lens, _, _, origin_articles,
       origin_abstracts, article_ids) = self._batch_reader.NextBatch()
      for i in xrange(self._hps.batch_size):
        bs = beam_search.BeamSearch(
            self._model, self._hps.batch_size,
            self._vocab.WordToId(data.START_DECODING),
            self._vocab.WordToId(data.STOP_DECODING),
            self._hps.dec_timesteps)

        article_batch_cp = article_batch.copy()
        article_batch_cp[:] = article_batch[i:i+1]
        article_lens_cp = article_lens.copy()
        article_lens_cp[:] = article_lens[i:i+1]
        article_ids_cp = article_ids.copy()
        article_ids_cp[:] = article_ids[i:i+1]
        best_beam = bs.run_beam_search(sess, article_batch_cp, article_lens_cp)
        decode_output = [int(t) for t in best_beam.tokens[1:]]
        self._decode_batch(
            origin_articles[i], origin_abstracts[i], decode_output, article_ids[i])
    return True

  def _decode_batch(self, article, abstract, output_ids, article_id):
    """Convert id to words and writing results.

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      output_ids: The abstract word ids output by machine.
    """
    decoded_output = ' '.join(data.ids2words(output_ids, self._vocab))
    end_p = decoded_output.find(data.SENTENCE_END, 0)
    if end_p != -1:
      decoded_output = decoded_output[:end_p]
    tf.logging.info('id:  %s', article_id)
#     tf.logging.info('article:  %s', article)
    tf.logging.info('abstract: %s', abstract)
    tf.logging.info('decoded:  %s', decoded_output)
    self._decode_io.write(abstract, decoded_output.strip(), article, article_id)
    
    
  def write_for_rouge(self, reference_sents, decoded_words, ex_index):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
      try:
        fst_period_idx = decoded_words.index(".")
      except ValueError: # there is text remaining that doesn't end in "."
        fst_period_idx = len(decoded_words)
      sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
      decoded_words = decoded_words[fst_period_idx+1:] # everything else
      decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = decoded_words # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    tf.logging.info('Wrote visualization data to %s', output_fname)


# def print_results(article, abstract, decoded_output):
#   """Prints the article, the reference summmary and the decoded summary to screen"""
#   print ""
#   tf.logging.info('ARTICLE:  %s', article)
#   tf.logging.info('REFERENCE SUMMARY: %s', abstract)
#   tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
#   print ""
# 
# 
# def make_html_safe(s):
#   """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
#   s.replace("<", "&lt;")
#   s.replace(">", "&gt;")
#   return s
# 
# 
# def rouge_eval(ref_dir, dec_dir):
#   """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
#   r = pyrouge.Rouge155()
#   r.model_filename_pattern = '#ID#_reference.txt'
#   r.system_filename_pattern = '(\d+)_decoded.txt'
#   r.model_dir = ref_dir
#   r.system_dir = dec_dir
#   logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
#   rouge_results = r.convert_and_evaluate()
#   return r.output_to_dict(rouge_results)

