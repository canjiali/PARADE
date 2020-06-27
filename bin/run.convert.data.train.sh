#!/bin/bash -xe

export max_num_train_instance_perquery=500
export rerank_threshold=2000
export BERT_PRETRAINED_DIR="gs://cloud-tpu-checkpoints/bert"
export num_segment=32

#for fold in {1..5}
#do
#export output_dir=/data2/robust04/runs/title/training.data/num-segment-${num_segment}/fold-${fold}-train-${max_num_train_instance_perquery}-test-${rerank_threshold}
export output_dir=/data-crystina/canjia/data/covid/training.data/num-segment-${num_segment}/round3/train-${max_num_train_instance_perquery}-test-${rerank_threshold}
mkdir -p $output_dir 

python3 -u generate_data.py \
    --trec_run_filename=/data-crystina/canjia/data/covid/runs/anserini.covid-r2.fusion2.txt \
    --qrels_filename=/data-crystina/canjia/tool/anserini/src/main/resources/topics-and-qrels/qrels.covid-round2-cumulative.txt \
    --query_filename=/data-crystina/canjia/tool/anserini/src/main/resources/topics-and-qrels/topics.covid-round2.xml \
    --query_field=question \
    --corpus_filename=/data-crystina/canjia/data/covid/runs/anserini.covid-r2.fusion2.txt.docid_content.txt \
    --output_dir=$output_dir \
    --dataset=covid19_r2 \
    --fold=1 \
    --vocab_filename=${BERT_PRETRAINED_DIR}/uncased_L-24_H-1024_A-16/vocab.txt \
    --plen=150 \
    --overlap=50 \
    --max_seq_length=256 \
    --max_num_segments_perdoc=${num_segment} \
    --max_num_train_instance_perquery=$max_num_train_instance_perquery \
    --rerank_threshold=$rerank_threshold \
    --do_fold_training=False \
    --convert_train=True \
    --convert_test=False \
    --used_qid_list=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35
