
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np
import tensorflow.compat.v1 as tf

from parade import Parade
from input_parser import input_fn_builder
from utils import result_info, relevance_info
from utils.fold_config import FOLD_CONFIG_DICT
from bert import optimization as bert_optimization
from electra import optimization as electra_optimization

tf.random.set_random_seed(118)
np.random.seed(118)

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_boolean(
  "from_distilled_student", False,
  "whether the ckpt comes from distilled student"
)
flags.DEFINE_string(
  "pretrained_model", 'bert',
  "which pretrained model: bert, electra"
)

flags.DEFINE_boolean(
  "use_passage_pos_embedding", False,
  "whether to use passage level position embedding"
)

flags.DEFINE_string(
  "aggregation_method", None,
  "which method for relevance aggregation. "
  "Support the following: 'cls-maxp', 'cls-avgp', 'cls-independent', 'cls-transformer'"
)

flags.DEFINE_integer(
  "CLS_ID", 101,
  "ID for merging the segments CLS embeddings. "
  "100: [UNK]; 101: [CLS]; 102: [SEP]; 103: [MASK]"
)

flags.DEFINE_integer(
  "num_transformer_layers", 2,
  "how many transformer layers for aggregation"
)

flags.DEFINE_string(
    "trec_run_filename", None,
    "where the trec run file (e.g. produced by BM25) is"
)

flags.DEFINE_string(
    "qrels_filename", None,
    "where the qrels file is"
)

flags.DEFINE_string(
    "dataset", None,
    "which dataset to run on. it would correspond to the fold config of qids"
)

flags.DEFINE_integer(
    "fold", 3,
    "run fold")

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

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_num_segments_perdoc", 8,
    "The maximum number of segments for each document"
)

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 3,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("num_train_steps", None,
                   "Total number of training steps to perform. "
                   "If this is set, the argument 'num_train_epochs' takes no effect")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


if FLAGS.pretrained_model == 'bert':
  from bert import modeling
elif FLAGS.pretrained_model == 'electra':
  from electra import modeling
else:
  raise ValueError("Unsupport model: {}".format(FLAGS.pretrained_model))


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, num_segments,
                 aggregation_method,
                 pretrained_model='bert', from_distilled_student=False,):
  """Creates a classification model."""
  scope = ""
  if from_distilled_student:
    scope = "student"
  parade_model = Parade(
    bert_config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    num_segments=num_segments,
    pretrained_model=pretrained_model,
    use_one_hot_embeddings=use_one_hot_embeddings,
    scope=scope
  )
  output_layer = None
  if aggregation_method == 'cls_attn':
    output_layer = parade_model.reduced_by_attn()
  elif aggregation_method == 'cls_avg':
    output_layer = parade_model.reduced_by_avg()
  elif aggregation_method == 'cls_max':
    output_layer = parade_model.reduced_by_max()
  elif aggregation_method == 'cls_transformer':
    output_layer = parade_model.reduced_by_transformer(is_training, num_transformer_layers=2)
  else:
    raise ValueError("Un-supported model type: {}".format(aggregation_method))

  with tf.variable_scope(scope):
    output_weights = tf.get_variable(
      "output_weights", [num_labels, parade_model.hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    logits = tf.tensordot(output_layer, output_weights, axes=[-1, -1])
    logits = tf.nn.bias_add(logits, output_bias)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, aggregation_method,
                     pretrained_model, from_distilled_student):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label"]
    num_segments = features["num_segments"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    (total_loss, per_example_loss, log_probs) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings, num_segments, aggregation_method,
        pretrained_model, from_distilled_student)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    initialized_variable_names = []
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    optimization_dict = {
      'bert': bert_optimization,
      'electra': electra_optimization
    }
    optimization = optimization_dict[pretrained_model]
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps,
          num_warmup_steps=num_warmup_steps, use_tpu=use_tpu
      )
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "log_probs": log_probs,
              "label_ids": label_ids,
          },
          scaffold_fn=scaffold_fn)

    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `FLAGS.do_train` or `FLAGS.do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_filename)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # training config
  qid_list = FOLD_CONFIG_DICT[FLAGS.dataset]
  qid_list = collections.deque(qid_list)
  rotate = FLAGS.fold - 1
  map(qid_list.rotate(rotate), qid_list)
  train_qid_list, test_qid_list = qid_list[0] + qid_list[1] + qid_list[2] + qid_list[3], qid_list[4]
  train_qid_list, test_qid_list = sorted(train_qid_list), sorted(test_qid_list)
  relevance_dict = relevance_info.create_relevance(FLAGS.trec_run_filename, FLAGS.qrels_filename)
  tf.logging.info("Running on dataset: {0}, on fold {1}".format(FLAGS.dataset, FLAGS.fold))
  tf.logging.info("Traing on the following qids: {0}\n".format(train_qid_list))
  tf.logging.info("Testing on the following qids: {0}\n".format(test_qid_list))
  if FLAGS.num_train_steps:
    num_train_steps = FLAGS.num_train_steps
  else:
    # we assume each query has the maximum number of training instances, this should not be painful
    num_train_queries = len(train_qid_list)
    num_train_steps = FLAGS.num_train_epochs * num_train_queries * FLAGS.max_num_train_instance_perquery
    num_train_steps = num_train_steps / FLAGS.train_batch_size
    # we'd also like it to be a multiple of thousands
    num_train_steps = int(num_train_steps//1000*1000)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  tf.logging.info("Number of training steps: {}".format(num_train_steps))
  tf.logging.info("Number of warmup steps: {}".format(num_warmup_steps))

  tf.gfile.MakeDirs(FLAGS.output_dir)
  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=1,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=2,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      aggregation_method=FLAGS.aggregation_method,
      pretrained_model=FLAGS.pretrained_model,
      from_distilled_student=FLAGS.from_distilled_student
  )
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = input_fn_builder(
      dataset_path=[os.path.join(FLAGS.data_dir, "dataset_train.tfrecord")],
      max_num_segments_perdoc=FLAGS.max_num_segments_perdoc,
      max_seq_length=FLAGS.max_seq_length,
      is_training=True)

    estimator.train(input_fn=train_input_fn,
                    max_steps=num_train_steps)
    tf.logging.info("Done Training!")

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation on the test set*****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
      dataset_path=[os.path.join(FLAGS.data_dir, "dataset_test.tfrecord")],
      max_num_segments_perdoc=FLAGS.max_num_segments_perdoc,
      max_seq_length=FLAGS.max_seq_length,
      is_training=False)

    trec_file = os.path.join(FLAGS.output_dir, "fold_{}_epoch_{}_bert_predictions_test.txt".format(FLAGS.fold, FLAGS.num_train_epochs))

    tf.logging.set_verbosity(tf.logging.WARN)
    result = estimator.predict(input_fn=eval_input_fn,
                               yield_single_examples=True)
    results = []
    for item in result:
      results.append(
        (item["log_probs"], item["label_ids"]))
    log_probs, labels = zip(*results)
    log_probs = np.stack(log_probs).reshape(-1, 2)
    scores = log_probs[:, 1]
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("Number of probs: {}".format(len(log_probs)))

    result_info.write_result_from_score(
      rerank_topk=FLAGS.rerank_threshold,
      scores=scores,
      qid_list=test_qid_list,
      relevance_dict=relevance_dict,
      write_path=trec_file,
      runid=FLAGS.aggregation_method
    )


if __name__ == "__main__":
  tf.app.run()

