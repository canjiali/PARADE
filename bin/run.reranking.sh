#!/bin/bash -xe

# things to take care of: tpu_name, method, runid
export method=cls_transformer
export tpu_name=node-zhi1
export field=question
export dataset=covid19_r2
export num_segment=32
export max_num_train_instance_perquery=500
export rerank_threshold=2000
export learning_rate=3e-6
export epoch=3

#export BERT_ckpt=gs://Your_gs_bucket/checkpoint/bertbase_msmarco/bert_model.ckpt
export BERT_config=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json 
export BERT_ckpt=gs://canjiampii/checkpoint/vanilla_electra_base_on_MSMARCO/model.ckpt-400000
export runid=electra_base_onMSMARCO_${method}_train500

#export data_dir="gs://canjiampii/adhoc/training.data/$dataset/$field/num-segment-${num_segment}/round3"
#export output_dir="gs://canjiampii/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/$runid/fold-$fold"
export data_dir=gs://canjiampii/adhoc/covid/training.data/num-segment-${num_segment}/round3/train-${max_num_train_instance_perquery}-test-${rerank_threshold}
export output_dir=gs://canjiampii/adhoc/covid/experiment/num-segment-${num_segment}/round3/${runid}


python3 -u run_reranking.py \
    --pretrained_model=electra \
    --do_train=True \
    --do_eval=True \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --learning_rate=$learning_rate \
    --num_train_steps=6000 \
    --num_train_epochs=$epoch \
    --warmup_proportion=0.1 \
    --aggregation_method=$method \
    --dataset=$dataset \
    --trec_run_filename=/data-crystina/canjia/data/covid/runs/anserini.covid-r3.fusion2.txt \
    --bert_config_filename=${BERT_config} \
    --init_checkpoint=${BERT_ckpt} \
    --data_dir=$data_dir \
    --output_dir=$output_dir \
    --max_seq_length=256 \
    --max_num_segments_perdoc=${num_segment} \
    --max_num_train_instance_perquery=$max_num_train_instance_perquery \
    --rerank_threshold=$rerank_threshold \
    --do_fold_training=False \
    --used_qid_list=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 \
    --use_tpu=True \
    --tpu_name=$tpu_name 
delete_tpu ${tpu_name}

gs_dir=gs://canjiampii/adhoc/covid/experiment/num-segment-${num_segment}/round3/${runid}
local_dir=/data-crystina/canjia/data/covid/reruns/round3/${runid}
qrels_path=/data-crystina/canjia/tool/anserini/src/main/resources/topics-and-qrels/qrels.covid-round3.txt
filter_qrels_path=/data-crystina/canjia/tool/anserini/src/main/resources/topics-and-qrels/qrels.covid-round2-cumulative.txt

./bin/download_and_evaluate.sh  ${gs_dir} ${local_dir} ${qrels_path} ${epoch} ${filter_qrels_path}
