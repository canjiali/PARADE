#!/bin/bash -xe

# things to take care of: tpu_name, method, runid
export method=cls_transformer
export tpu_name=node-zhi2
export field=title
export dataset=robust04
export num_segment=16
export max_num_train_instance_perquery=1000
export rerank_threshold=100
export learning_rate=3e-6
export epoch=1

export BERT_config=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json
export BERT_ckpt=gs://canjiampii/experiment/vanilla_electra_base_onMSMARCO/model.ckpt-400000
#export BERT_ckpt=gs://canjiampii/checkpoint/bertbase_msmarco/bert_model.ckpt
export runid=electra_base_onMSMARCO_${method}

for fold in {1..5}
do
  export data_dir="gs://canjiampii/adhoc/training.data/$dataset/$field/num-segment-${num_segment}/fold-$fold-train-$max_num_train_instance_perquery-test-$rerank_threshold"
  export output_dir="gs://canjiampii/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/$runid/fold-$fold"

  python3 -u run_ranking_v2.py \
    --pretrained_model=electra \
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
    --bert_config_filename=${BERT_config} \
    --init_checkpoint=${BERT_ckpt} \
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

gs_dir=gs://canjiampii/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/$runid
local_dir=/data2/$dataset/reruns/$field/num-segment-${num_segment}/$runid
qrels_path=/data/anserini/src/main/resources/topics-and-qrels/qrels.${dataset}.txt

./bin/download_and_evaluate.sh ${gs_dir} ${local_dir} ${qrels_path} ${epoch}