#!/bin/bash -xe

# things to take care of: tpu_name, method, runid
export method=cls_transformer
export field=title
export dataset=robust04
export num_segment=16
export max_num_train_instance_perquery=1000
export rerank_threshold=100
export learning_rate=3e-6
export epoch=3


student_BERT_config=gs://Your_gs_bucket/checkpoint/uncased_L-10_H-768_A-12_medium_10_base/bert_config.json
teacher_BERT_config=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json
student_BERT_ckpt_parent=gs://Your_gs_bucket/checkpoint/bert_small_onRobust04
teacher_BERT_ckpt_parent=gs://Your_gs_bucket/adhoc/experiment/robust04/title/num-segment-16/bertbase_onMSMARCO_cls_transformer


#MSE CE 
kd_method=MSE
kd_lambda=0.75
runid=base_to_small_kd-${kd_method}_lambda-${kd_lambda}_lr-${learning_rate}

for fold in {1..5}
do
  export data_dir="gs://Your_gs_bucket/adhoc/training.data/$dataset/$field/num-segment-${num_segment}/fold-$fold-train-$max_num_train_instance_perquery-test-$rerank_threshold"
  export output_dir="gs://Your_gs_bucket/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/KD/$runid/fold-$fold"
  export teacher_BERT_ckpt=${student_BERT_parent}/fold-${fold}/model.ckpt-18000
  export student_BERT_ckpt=${student_BERT_parent}/fold-${fold}/final

  python3 -u run_knowledge_distill.py \
    --pretrained_model=bert \
    --kd_method=${kd_method} \
    --kd_lambda=${kd_lambda} \
    --teacher_bert_config_file=${teacher_BERT_config} \
    --student_bert_config_file=${student_BERT_config} \
    --teacher_init_checkpoint=${teacher_BERT_ckpt}\
    --student_init_checkpoint=${student_BERT_ckpt}\
    --do_train=True \
    --do_eval=True \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --learning_rate=$learning_rate \
    --num_train_epochs=$epoch \
    --warmup_proportion=0.1 \
    --aggregation_method=$method \
    --dataset=$dataset \
    --fold=$fold \
    --trec_run_filename=/data2/robust04/runs/title/run.robust04.title.bm25.txt \
    --qrels_filename=/data/anserini/src/main/resources/topics-and-qrels/qrels.robust04.txt \
    --data_dir=$data_dir \
    --output_dir=$output_dir \
    --max_seq_length=256 \
    --max_num_segments_perdoc=${num_segment} \
    --max_num_train_instance_perquery=$max_num_train_instance_perquery \
    --rerank_threshold=$rerank_threshold \
    --use_tpu=True \
    --tpu_name=$tpu_name 
done
delete_tpu ${tpu_name}

gs_dir=gs://Your_gs_bucket/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/KD/$runid
local_dir=/data2/$dataset/reruns/$field/num-segment-${num_segment}/KD/$runid
qrels_path=/data/anserini/src/main/resources/topics-and-qrels/qrels.${dataset}.txt

./scripts/download_and_evaluate.sh  ${gs_dir} ${local_dir} ${qrels_path} ${epoch} 
