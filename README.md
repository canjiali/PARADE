# PARADE

## For COVID-19 only
#### convert data
If you want to do a 5-fold cross validatoin, set `do_fold_training=True`;
otherwise you want to train on a list of qids and evaluate on a list of other qids (though there exist some overlap),
you set `do_fold_training=False` and configure the `used_qid_list` argument.
You're supposed to run this script twice, because for training and evaluation, 
the `trec_run_filename`, `corpus_filename` should be different, as well as the `used_qid_list`.
You set `convert_train=True` to convert training while `convert_test=True` for evaluation. 
DON'T do them at the same time.

#### run training and evaluation
Again if you want to do 5-fold cross validatoin, set `do_fold_training=True`;
otherwise you want to train on a list of qids and evaluate a list of other qids (though there exist some overlap),
you set `do_fold_training=False` and configure the `used_qid_list` argument.
The `used_qid_list` should be the same as your `used_qid_list` when you're converting evaluation data.

Both scripts are in `scripts/run.convert.data.sh` and `scripts/run.reranking.sh`.


## Introduction
PARADE (PAssage Representation Aggregation for Document rE-ranking) is a document re-ranking model based on the pre-trained language models.
This repo contains the code to reproduce PARADE.

We support the following PARADE variants:
- PARADE-AvgP (named `cls_avgP` in the code)
- PARADE-wAvgP (named `cls_wAvgP` in the code)
- PARADE-MaxP (namded `cls_maxP` in the code)
- PARADE-Transformer (namded `cls_transformer` in the code)

We support two instantiations of pre-trained models:
- BERT
- ELECTRA

## Getting Started
To run PARADE, there're three steps ahead.
We give a detailed example on how to run the code the Robust04 dataset using the title query.

### 1. Data Preparation
We need to split the documets into passages, write them into TFrecord files.
Data for 5 folds are required here.
The standard `qrels`, `query`, `trec_run` files can be accomplished by [Anserini](https://github.com/castorini/anserini),
please check out their notebook for further details.
The `corpus` file can also be extarcted by Anserini to form the `docno \t content` paired text.

```bash
export max_num_train_instance_perquery=1000
export rerank_threshold=100
export BERT_PRETRAINED_DIR="gs://cloud-tpu-checkpoints/bert"
export num_segment=16

for fold in {1..5}
do
  export output_dir="/data2/robust/runs/title/training.data/num-segment-${num_segment}/fold-${fold}-train-${max_num_train_instance_perquery}-test-${rerank_threshold}
  mkdir -p $output_dir 

  python3 generate_data.py \
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
```
You should be able to see 5 sub-folders generated in the `output_dir` folder,
with each contains a train file and a test file.
Note that if you're going to run the code on TPU, you need to upload the training/testing data to Google Cloud Storage(GCS).

The above snippet also exists in `scripts/run.convert.data.sh`.

If you bother getting the raw text from Anserini, 
you can also replace the `anserini/src/main/java/io/anserini/index/IndexUtils.java` file by the `extra/IndexUtils.java` file in this repo,
then re-build Anserini.
Below is how we fetch the raw text
```bash
anserini_path="path_to_anserini"
index_path="path_to_index"
# say you're given a BM25 run file run.BM25.txt
cut -d ' ' -f3 run.BM25.txt | sort | uniq > docnolist
${anserini_path}/target/appassembler/bin/IndexUtils -dumpTransformedDocBatch docnolist -index ${index_path}
```
then you get the required raw text in the same directory of docnolist. 
Everything is prepared now!

### 2. Model Traning and Evaluation



For all the pra-trained models, we first fine-tune them on the MSMARCO passage collection.
This is IMPORTANT, as it can improve the nDCG@20 by 2 points generally.
To figure out the way of doing that, please check out [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert).
The fine-tuned model will be the initialized model in PARADE.
Just pass it to the `BERT_ckpt` argument in the following snippet. 

For running 5-fold cross validation, you can check the fold configuration here at `utils/fold_config.py`.
```bash
export method=cls_transformer
export tpu_name=node-5
export field=title
export dataset=robust04
export num_segment=16
export max_num_train_instance_perquery=1000
export rerank_threshold=100
export learning_rate=3e-6
export epoch=3

export BERT_config=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json 
export BERT_ckpt="Your_path_to_the_pretrain_ckpt"
export runid=electra_base_onMSMARCO_${method}

for fold in {1..5}
do
  export data_dir="gs://Your_gs_bucket/adhoc/training.data/$dataset/$field/num-segment-${num_segment}/fold-$fold-train-$max_num_train_instance_perquery-test-$rerank_threshold"
  export output_dir="gs://Your_gs_bucket/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/$runid/fold-$fold"

  python3 -u run_reranking.py \
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
```
The 5-fold training/testing are done now.
Next you need to download the resutls from GCS, then merge them, and evaluate them!
```bash
gs_dir=gs://Your_gs_bucket/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/$runid
local_dir=/data2/$dataset/reruns/$field/num-segment-${num_segment}/$runid
qrels_path=/data/anserini/src/main/resources/topics-and-qrels/qrels.${dataset}.txt
mkdir -p $local_dir

# download 5-fold resutls from GCS
# if you're in a local machine, you don't need to download
for fold in {1..5}
do
  gsutil cp $gs_dir/fold-${fold}/fold_*epoch_${epoch}_bert_predictions_test.txt $local_dir
done

# make sure that we find out all the 5 result files
num_result=$(ls $local_dir |wc -l)
if [ "$num_result" != "5" ]; then
  echo Exit. Wrong number of results. Expect 5 but  $num_result result files found!
  exit
fi

# evaluate the result using trec_eval
cat ${local_dir}/fold_*epoch_${epoch}_bert_predictions_test.txt >> ${local_dir}/merge_epoch${epoch}
/data/tool/trec_eval-9.0.7/trec_eval ${qrels_path} ${local_dir}/merge_epoch${epoch} -m ndcg_cut.20 -m P.20  >> ${local_dir}/result_epoch${epoch}
cat ${local_dir}/result_epoch${epoch}
```
The model performance will automatically output on your screen. On Robust04 title using PARADE(ELECTRA), we get 
```
P_20                    all     0.4604
ndcg_cut_20             all     0.5399
```
Looks good!
The above steps can also be done all at once by running `scripts/run.reranking.sh`.

### 3. Significance Test
To do a significance test, just configurate the `trec_eval` path in the `evaluation.py` file. 
Then simply run the following command, here we compare PARADE with BERT-MaxP:
```
python evaluation.py \
  --qrels /data/anserini/src/main/resources/topics-and-qrels/qrels.robust04.txt \
  --baselines /data2/robust04/reruns/title/bertmaxp.dai/bertbase_onMSMARCO/merge \
  --runs /data2/robust04/reruns/title/num-segment-16/electra_base_onMSMARCO_cls_transformer/merge_epoch3
```
then it outputs
```
OrderedDict([('P_20', '0.4277'), ('ndcg_cut_20', '0.4931')])
OrderedDict([('P_20', '0.4604'), ('ndcg_cut_20', '0.5399')])
OrderedDict([('P_20', 1.2993259211924425e-11), ('ndcg_cut_20', 8.306604295574242e-09)])
```
The upper two lines are the sanity checks of your run performance values.
The last line represents the p-values.
PARADE achieves significant improvement over BERT-MaxP (p < 0.01) !

### Knowledge Distillation (Optional)
You can also perform knowledge distillation for the smaller PARADE models.
Please follow the above steps to fine-tune the smaller models first.
Then run the following command:
```bash
export method=cls_transformer
export field=title
export dataset=robust04
export num_segment=16
export max_num_train_instance_perquery=1000
export rerank_threshold=100
export learning_rate=3e-6
export epoch=3

student_BERT_config=gs://Your_gs_bucket/checkpoint/uncased_L-4_H-512_A-8_small/bert_config.json
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

gs_dir=gs://Your_gs_bucket/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/KD/$runid
local_dir=/data2/$dataset/reruns/$field/num-segment-${num_segment}/KD/$runid
qrels_path=/data/anserini/src/main/resources/topics-and-qrels/qrels.${dataset}.txt

./scripts/download_and_evaluate.sh  ${gs_dir} ${local_dir} ${qrels_path} ${epoch} 
```
The script also lies in `scripts/run.kd.sh`.
It outputs the following results for PARADE using BERT-small with only 4 layers!
```bash
P_20                    all     0.4365
ndcg_cut_20             all     0.5098
```