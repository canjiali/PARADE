
import os
import random
import collections
import tensorflow.compat.v1 as tf
import numpy as np

from bert import tokenization
from utils import file_operation, relevance_info
from utils.fold_config import FOLD_CONFIG_DICT

random.seed(118)
tf.random.set_random_seed(118)


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "trec_run_filename", None,
    "where the trec run file (e.g. produced by BM25) is"
)

flags.DEFINE_string(
    "qrels_filename", None,
    "where the qrels file is"
)

flags.DEFINE_string(
    "query_field", 'title',
    "None if no field, else title, desc, narr, question")

flags.DEFINE_string(
    "query_filename", None,
    "where the query file is. support TREC file now")

flags.DEFINE_string(
    "corpus_filename", None,
    "where the corpus file is. format: docno \t content")

flags.DEFINE_string(
    "dataset", None,
    "which dataset to run on. it would correspond to the fold config of qids"
)

flags.DEFINE_integer(
    "fold", 3,
    "run fold")

flags.DEFINE_integer(
    "plen", 150,
    "length of segmented passage"
)

flags.DEFINE_integer(
    "overlap", 50,
    "overlap between continuous segmented passages"
)

flags.DEFINE_integer(
  "max_num_train_instance_perquery", 1000,
  "The maximum number of training instances utilized from initial ranking"
)

flags.DEFINE_integer(
  "rerank_threshold", 100,
  "the maximum number of top documents to be reranked"
)

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_filename", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_filename", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_num_segments_perdoc", 8,
    "The maximum number of segments for each document"
)


# The average length of robust is 689

class PointwiseInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, exampleid, tokens_a, tokens_b_list, relation_label):
    self.exampleid = exampleid
    self.tokens_a = tokens_a
    self.tokens_b_list = tokens_b_list
    self.relation_label = relation_label

  def __str__(self):
    s = ""
    s += "example id: %s\n" % self.exampleid
    s += "tokens a: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens_a]))
    s += "tokens b: %s\n" % (" ".join(
      [tokenization.printable_text(x) for x in self.tokens_b_list]))
    s += "relation label: %s\n" % self.relation_label
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
  return feature


def convert_data_pointwise(writer, tokenizer, qid_list, relevance_dict, corpus_dict, query_dict, is_eval=False):
  if is_eval:
    max_num_example = FLAGS.rerank_threshold
  else:
    max_num_example = FLAGS.max_num_train_instance_perquery

  instances = []
  idx = 0
  for qid in qid_list:
    tf.logging.info("Generating data for query {}".format(qid))
    relevance = relevance_dict.get(qid)
    judged_docno_list = relevance.get_judged_docno_list()
    supervised_docno_list = relevance.get_supervised_docno_list() # initial ranking

    # training data from the judged docno,  built from bm25 top1000 result
    relevant_docno_list = set()
    if judged_docno_list is not None:
      relevant_docno_list = judged_docno_list[1] + judged_docno_list[2]
      relevant_docno_list = set(relevant_docno_list)
    for docno in supervised_docno_list[:max_num_example]:
      relation_label = 1 if docno in relevant_docno_list else 0
      query = query_dict[qid]
      doc = corpus_dict[docno]
      instance = create_instance_pointwise(tokenizer, FLAGS.max_seq_length, qid, docno, query, doc, relation_label)
      # append and shuffle on training set
      if not is_eval:
        instances.append(instance)
      else:
        write_instance_to_example_files(writer, tokenizer, instance, idx)
      idx += 1
  tf.logging.info("Totally {} examples".format(len(instances)))

  if not is_eval:
    random.shuffle(instances)
    for idx, instance in enumerate(instances):
      write_instance_to_example_files(writer, tokenizer, instance, idx)
  if is_eval:
    write_padding_instance_to_example_files(writer)
  writer.close()
  print("Distribution of length. Key is length, Val is count.")
  for key, val in stats.items():
    print("{}\t{}".format(key, val))

def create_instance_pointwise(tokenizer, max_seq_length, qid, docno, query, doc, label):
  query = tokenization.convert_to_unicode(query)
  doc = tokenization.convert_to_unicode(doc)
  passages = get_passages(doc, FLAGS.plen, FLAGS.overlap)
  if len(passages) == 0:
    tf.logging.warn("Passage length is 0 in qid {} docno {}".format(qid, docno))

  query = tokenization.convert_to_bert_input(
    text=query,
    max_seq_length=64,
    tokenizer=tokenizer,
    add_cls=True,
    convert_to_id=False
  )
  passages = [tokenization.convert_to_bert_input(
    text=p,
    max_seq_length=max_seq_length-len(query),
    tokenizer=tokenizer,
    add_cls=False,
    convert_to_id=False
  ) for p in passages]
  instance = PointwiseInstance(
    exampleid="{}-{}".format(qid, docno),
    tokens_a=query,
    tokens_b_list=passages,
    relation_label=label
  )

  return instance


def write_padding_instance_to_example_files(writer, num_examples=50):
  # 1-d arrays
  input_ids = np.zeros((FLAGS.max_seq_length * FLAGS.max_num_segments_perdoc), dtype=np.int)
  num_segments = FLAGS.max_num_segments_perdoc
  label = 0
  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["tokens_a_len"] = create_int_feature([3])
  features["tokens_ids_lens"] = create_int_feature([24] * FLAGS.max_num_segments_perdoc)
  features["num_segments"] = create_int_feature([num_segments])
  features["label"] = create_int_feature([label])
  tf_example = tf.train.Example(features=tf.train.Features(feature=features))

  for _ in range(num_examples):
    writer.write(tf_example.SerializeToString())
  tf.logging.info("write {} padding instances successfully".format(num_examples))


def write_instance_to_example_files(writer, tokenizer, instance, instance_idx):

  def padding_2d(ids_list, num_tokens_per_segment, padding_value=0):
    _len = len(ids_list)
    if padding_value == 0:
      matrix = np.zeros((_len, num_tokens_per_segment), dtype=np.int)
    elif padding_value == 1:
      matrix = np.ones((_len, num_tokens_per_segment), dtype=np.int)
    else:
      raise ValueError("Unsupport padding value")

    for i, _list in enumerate(ids_list):
      matrix[i, :len(_list)] = _list

    return matrix.flatten()

  tokens_a = instance.tokens_a
  tokens_b_list = instance.tokens_b_list
  tokens_a_ids = tokenizer.convert_tokens_to_ids(tokens_a)
  tokens_b_list = [tokenizer.convert_tokens_to_ids(p) for p in tokens_b_list]
  label = instance.relation_label
  assert len(tokens_b_list) <= FLAGS.max_num_segments_perdoc
  num_segments = len(tokens_b_list)

  input_ids = [tokens_a_ids + tokens_b_passage_ids for tokens_b_passage_ids in tokens_b_list]
  tokens_a_len = len(tokens_a_ids)  # helpful for segment ids
  input_ids_lens = [len(input_id) for input_id in input_ids]  # helpful for input mask
  input_ids_lens = input_ids_lens + [FLAGS.max_seq_length] * (FLAGS.max_num_segments_perdoc - len(input_ids_lens))
  input_ids = padding_2d(input_ids,FLAGS.max_seq_length, padding_value=0)
  # write to tfrecord
  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["tokens_a_len"] = create_int_feature([tokens_a_len])
  features["tokens_ids_lens"] = create_int_feature(input_ids_lens)
  features["num_segments"] = create_int_feature([num_segments])
  features["label"] = create_int_feature([label])
  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  writer.write(tf_example.SerializeToString())

  if instance_idx < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("tokens_a: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens_a]))
    tf.logging.info("tokens_b_list: {}".format(instance.tokens_b_list))

    for feature_name in features.keys():
      feature = features[feature_name]
      values = []
      if feature.int64_list.value:
        values = feature.int64_list.value
      elif feature.float_list.value:
        values = feature.float_list.value
      tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

stats = collections.defaultdict(int)
def get_passages(text, plen, overlap):
    """ Modified from https://github.com/AdeDZY/SIGIR19-BERT-IR/blob/master/tools/gen_passages.py
    :param text:
    :param plen:
    :param overlap:
    :return:
    """
    words = text.strip().split(' ')
    s, e = 0, 0
    passages = []
    while s < len(words):
      e = s + plen
      if e >= len(words):
        e = len(words)
      # if the last one is shorter than 'overlap', it is already in the previous passage.
      if len(passages) > 0 and e - s <= overlap:
        break
      p = ' '.join(words[s:e])
      passages.append(p)
      s = s + plen - overlap

    if len(passages) > FLAGS.max_num_segments_perdoc:
      chosen_ids = sorted(random.sample(range(1, len(passages) - 1), FLAGS.max_num_segments_perdoc - 2))
      chosen_ids = [0] + chosen_ids + [len(passages) - 1]
      passages = [passages[id] for id in chosen_ids]

    global stats
    stats[len(passages)] += 1
    return passages


def main(_):
  # training config
  qid_list = FOLD_CONFIG_DICT[FLAGS.dataset]
  qid_list = collections.deque(qid_list)
  rotate = FLAGS.fold - 1
  map(qid_list.rotate(rotate), qid_list)

  # currently, we just set up the training step. No support for model selection now.
  # train_qid_list, valid_qid_list, test_qid_list = qid_list[0] + qid_list[1] + qid_list[2], qid_list[3], qid_list[4]
  train_qid_list, test_qid_list = qid_list[0] + qid_list[1] + qid_list[2] + qid_list[3], qid_list[4]
  train_qid_list, test_qid_list = sorted(train_qid_list), sorted(test_qid_list)
  tf.logging.info("Running on dataset: {0}, on fold {1}".format(FLAGS.dataset, FLAGS.fold))
  tf.logging.info("Traing on following qid: {0}\n".format(train_qid_list))
  # tf.logging.info("Validating on following qid: {0}\n".format(valid_qid_list))
  tf.logging.info("Testing on following qid: {0}\n".format(test_qid_list))

  relevance_dict = relevance_info.create_relevance(FLAGS.trec_run_filename, FLAGS.qrels_filename)
  corpus_dict = file_operation.key_value_from_file(FLAGS.corpus_filename)
  query_dict = file_operation.load_trec_topics(FLAGS.query_filename)[FLAGS.query_field]
  tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_filename,
    do_lower_case=FLAGS.do_lower_case
  )
  # begin data convertion to TFrecord
  output_path = os.path.join(FLAGS.output_dir, "dataset_train.tfrecord")
  tf.logging.info("Writing data into {}".format(output_path))
  writer = tf.python_io.TFRecordWriter(output_path)
  convert_data_pointwise(
    writer=writer,
    tokenizer=tokenizer,
    qid_list=train_qid_list,
    relevance_dict=relevance_dict,
    corpus_dict=corpus_dict,
    query_dict=query_dict,
    is_eval=False
  )
  output_path = os.path.join(FLAGS.output_dir, "dataset_test.tfrecord")
  tf.logging.info("Writing data into {}".format(output_path))
  writer = tf.python_io.TFRecordWriter(output_path)
  convert_data_pointwise(
    writer=writer,
    tokenizer=tokenizer,
    qid_list=test_qid_list,
    relevance_dict=relevance_dict,
    corpus_dict=corpus_dict,
    query_dict=query_dict,
    is_eval=True
  )


if __name__ == '__main__':
  flags.mark_flag_as_required("trec_run_filename")
  flags.mark_flag_as_required("qrels_filename")
  flags.mark_flag_as_required("query_field")
  flags.mark_flag_as_required("query_filename")
  flags.mark_flag_as_required("corpus_filename")
  flags.mark_flag_as_required("dataset")
  flags.mark_flag_as_required("fold")
  flags.mark_flag_as_required("vocab_filename")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("plen")
  flags.mark_flag_as_required("overlap")

  tf.app.run()
