
import tensorflow.compat.v1 as tf
from bert import modeling as bert_modeling
from electra import modeling as electra_modeling

modeling_dict = {
  'bert': bert_modeling,
  'electra': electra_modeling
}
class DocuBERT(object):
  def __init__(self, bert_config, is_training, input_ids, input_mask, segment_ids,
               num_segments, pretrained_model='bert', use_one_hot_embeddings= True,
               from_distilled_student=False):

    modeling = modeling_dict[pretrained_model]
    input_shape = modeling.get_shape_list(input_ids, expected_rank=3)
    batch_size, max_num_segments_perdoc, max_seq_length = input_shape

    # we reshap the input here because bert only support rank-2 matrices
    # i.e. [B, N, L] -> [B x N, L]
    input_ids_2d = modeling.reshape_to_matrix(input_ids)
    input_mask_2d = modeling.reshape_to_matrix(input_mask)
    segment_ids_2d = modeling.reshape_to_matrix(segment_ids)

    scope_prefix = ""
    if from_distilled_student:
      scope_prefix = 'student'
    model = modeling.BertModel(
      bert_config,
      is_training=is_training,
      input_ids=input_ids_2d,
      input_mask=input_mask_2d,
      token_type_ids=segment_ids_2d,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=scope_prefix + '/' + pretrained_model if len(scope_prefix) > 0 else pretrained_model
    )
    pooled_output_layer = model.get_pooled_output()  # [B x N, H]
    self.output_layer = modeling.reshape_from_matrix(pooled_output_layer, input_shape)  # [B, N, H]
    self.model = model
    self.modeling = modeling

    # related to segment level masks
    self.num_segments = num_segments
    self.segment_mask = tf.sequence_mask(num_segments, max_num_segments_perdoc, dtype=tf.float32)  # [B, N]
    self.adder = (1.0 - tf.cast(self.segment_mask, tf.float32)) * -10000.0
    # related to config
    self.bert_config = bert_config
    self.batch_size, self.max_num_segments_perdoc, self.max_seq_length = \
      batch_size, max_num_segments_perdoc, max_seq_length
    self.hidden_size = self.output_layer.shape[-1].value


  def reduced_by_wAvgP(self):
    cls_weight = tf.get_variable(
      'cls_weight', shape=[self.hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_layer = self.output_layer
    weights = tf.tensordot(output_layer, cls_weight, axes=[-1, -1])  # [B, N]
    weights = weights + self.adder
    weights = tf.nn.softmax(weights, axis=-1)
    output_layer = tf.reduce_sum(output_layer * tf.expand_dims(weights, axis=-1), axis=1)  # [B, H]

    return output_layer


  def reduced_by_avgP(self):
    output_layer = self.output_layer
    output_layer = tf.expand_dims(self.segment_mask, axis=-1) * output_layer
    output_layer = tf.reduce_sum(output_layer, axis=1)  # [B, H]
    # be careful, cannot be divied by zero (no segments)
    output_layer = tf.div(output_layer,
                          tf.cast(tf.expand_dims(self.num_segments, -1), tf.float32))

    return output_layer

  def reduced_by_maxP(self):
    output_layer = self.output_layer
    output_layer = tf.expand_dims(self.adder, axis=-1) + output_layer
    output_layer = tf.reduce_max(output_layer, axis=1)  # [B, H]

    return output_layer

  def reduced_by_transformer(self, is_training, num_transformer_layers= 2, CLS_ID=102,
                             use_passage_pos_embedding=False):
    bert_config = self.bert_config
    output_layer = self.output_layer
    model = self.model
    embeddings = model.get_embedding_table()
    # clsid_tf = tf.constant([CLS_ID], dtype=tf.int32, name="clsid_tf")
    clsid_tf = tf.Variable([CLS_ID], dtype=tf.int32, trainable= False, name='clsid_tf')
    cls_embedding = tf.nn.embedding_lookup(embeddings, clsid_tf)
    cls_embedding_tiled = tf.tile(cls_embedding, multiples=[self.batch_size, 1]) # [B, H]
    merged_output = tf.concat((output_layer, tf.expand_dims(cls_embedding_tiled, axis=1)), axis=1) # [B, N + 1, H]
    if use_passage_pos_embedding:
      full_position_embeddings = tf.get_variable(
        name="passage_position_embedding",
        shape=[self.max_num_segments_perdoc+1, self.hidden_size],
        initializer=self.modeling.create_initializer(0.02))
      full_position_embeddings = tf.expand_dims(full_position_embeddings, axis=0)
      merged_output += full_position_embeddings

    # here comes the Transformer.
    attention_mask = tf.sequence_mask(self.num_segments+1, self.max_num_segments_perdoc+1, dtype=tf.float32)
    attention_mask = tf.tile(tf.expand_dims(attention_mask, axis=1), [1, self.max_num_segments_perdoc+1, 1])
    with tf.variable_scope("docubert_transformer"):
      if not is_training:
        bert_config.hidden_dropout_prob = 0.0
        bert_config.attention_probs_dropout_prob = 0.0
      output_layer, _ = self.modeling.transformer_model(
        input_tensor=merged_output,
        attention_mask=attention_mask,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=num_transformer_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        hidden_dropout_prob=bert_config.hidden_dropout_prob,
        attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
        initializer_range=bert_config.initializer_range,
        do_return_all_layers=False
      ) # [B, N + 1, H]
      output_layer = tf.squeeze(output_layer[:, 0:1, :], axis=1) # [B, H]

    return output_layer
