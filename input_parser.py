
import tensorflow.compat.v1 as tf

def input_fn_builder(dataset_path, is_training, max_num_segments_perdoc, max_seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""

    batch_size = params["batch_size"]
    output_buffer_size = batch_size * 1000

    def extract_fn(data_record):
      features = {
        "input_ids": tf.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
        ),
        "tokens_a_len": tf.FixedLenFeature(
          [], tf.int64
        ),
        "tokens_ids_lens": tf.FixedLenFeature(
          [max_num_segments_perdoc], tf.int64
        ),
        "num_segments": tf.FixedLenFeature(
          [], tf.int64
        ),
        "label": tf.FixedLenFeature(
          [], tf.int64
        )
      }

      sample = tf.parse_single_example(data_record, features)
      tokens_a_len = sample.pop("tokens_a_len")
      tokens_ids_lens = sample.pop("tokens_ids_lens")
      # 0 0 0 ... 1 1 1 1 ...
      segment_ids = 1 - tf.sequence_mask(tokens_a_len, max_seq_length, dtype=tf.int32)
      segment_ids = tf.tile(tf.expand_dims(segment_ids, axis=0), multiples=[max_num_segments_perdoc, 1])
      # 1 1 1 1 ... 0 0 0 ...
      input_mask = tf.sequence_mask(tokens_ids_lens, max_seq_length, dtype=tf.int32)
      sample.update({
        "segment_ids": segment_ids,
        "input_mask": input_mask
      })
      sample["input_ids"] = tf.reshape(sample["input_ids"], shape=[-1, max_seq_length])

      # the extracted features are exactly what we want, no need for data convertion, hence return
      # before return, convert to tf.int32 for TPU
      for key, val in sample.items():
        sample[key] = tf.cast(sample[key], tf.int32)

      return sample


    dataset = tf.data.TFRecordDataset(dataset_path)
    dataset = dataset.map(
        extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=10000)
    # else:
    #   if max_eval_examples:
    #     dataset = dataset.take(max_eval_examples)

    dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes={
        "input_ids": [max_num_segments_perdoc, max_seq_length],
        "segment_ids": [max_num_segments_perdoc, max_seq_length],
        "input_mask": [max_num_segments_perdoc, max_seq_length],
        "num_segments": [],
        "label": [],
      },
      padding_values={
        "input_ids": 0,
        "segment_ids": 0,
        "input_mask": 0,
        "num_segments": 0,
        "label": 0,
      },
      drop_remainder=True)

    return dataset
  return input_fn