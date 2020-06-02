#!/bin/bash -xe

export max_num_train_instance_perquery=1000
export rerank_threshold=100
export BERT_PRETRAINED_DIR="gs://cloud-tpu-checkpoints/bert"
export num_segment=16

for fold in {1..5}
do
  export output_dir=/data2/robust04/runs/title/training.data/num-segment-${num_segment}/fold-${fold}-train-${max_num_train_instance_perquery}-test-${rerank_threshold}
  mkdir -p $output_dir 

  python3 -u generate_data.py \
    --trec_run_filename=/data2/robust04/runs/title/run.robust04.title.bm25.txt \
    --qrels_filename=/data/anserini/src/main/resources/topics-and-qrels/qrels.robust04.txt \
    --query_filename=/data/anserini/src/main/resources/topics-and-qrels/topics.robust04.txt \
    --query_field=title \
    --corpus_filename=/data2/robust04/runs/title/run.robust04.title.bm25.txt.docno.uniq_rawdocs.txt \
    --output_dir=$output_dir \
    --dataset=robust04 \
    --fold=$fold \
    --vocab_filename=${BERT_PRETRAINED_DIR}/uncased_L-24_H-1024_A-16/vocab.txt \
    --plen=150 \
    --overlap=50 \
    --max_seq_length=256 \
    --max_num_segments_perdoc=${num_segment} \
    --max_num_train_instance_perquery=$max_num_train_instance_perquery \
    --rerank_threshold=$rerank_threshold 
done
