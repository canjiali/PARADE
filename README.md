# transformers_ranking

## Getting Started
To run DocuBERT, there're three steps ahead.
We give a detailed example on Robust04 dataset using the title query.

### 1. Data Preparation
We need to split the documets into passages, write them into TFrecord files.
Data for 5 folds are required here.
The standard `qrels`, `query`, `trec_run` files can be fulfilled by [Anserini](https://github.com/castorini/anserini),
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
The above snippet also exists in `scripts/run.convert.data.sh`.
Note that if you're going to run the code on TPU, you need to upload the training/testing data to GCS.

### 2. Model Traning and Evaluation

We support the following DocuBERT variants:
- DocuBERT-AvgP (passed to the `method` as cls_avgP)
- DocuBERT-wAvgP (passed to the `method` as cls_wAvgP)
- DocuBERT-MaxP (passed to the `method` as cls_maxP)
- DocuBERT-Transformer (passed to the `method` as cls_transformer)

We support two pre-trained model instantiations:
- BERT
- ELECTRA

For all the pra-trained models, we first fine-tune them on the MSMARCO passage collection.
This is IMPORTANT, as it can improve the nDCG@20 by 2 points generally.
To figure out the way of doing that, please check out [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert).
The fine-tuned model will be the initialized model in DocuBERT.
Just pass it to the `BERT_ckpt` argument in the following snippet. 


```bash
export method=cls_transformer
export tpu_name=node-5
export field=title
export dataset=robust04
export num_segment=16
export max_num_train_instance_perquery=1000
export rerank_threshold=100
export learning_rate=3e-6
export epoch=1

#export BERT_ckpt=gs://canjiampii/checkpoint/bertbase_msmarco/bert_model.ckpt
export BERT_config=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json 
export BERT_ckpt=gs://canjiampii/experiment/vanilla_electra_base_onMSMARCO/model.ckpt-400000
export runid=electra_base_onMSMARCO_${method}

for fold in {1..5}
do
  export data_dir="gs://canjiampii/adhoc/training.data/$dataset/$field/num-segment-${num_segment}/fold-$fold-train-$max_num_train_instance_perquery-test-$rerank_threshold"
  export output_dir="gs://canjiampii/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/$runid/fold-$fold"

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
gs_dir=gs://canjiampii/adhoc/experiment/$dataset/$field/num-segment-${num_segment}/$runid
local_dir=/data2/$dataset/reruns/$field/num-segment-${num_segment}/$runid
qrels_path=/data/anserini/src/main/resources/topics-and-qrels/qrels.${dataset}.txt
mkdir -p $local_dir

# download 5-fold resutls from GCS
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
The model performance will automatically output on yor screen. On Robust04 title using DocuBERT-Transformer(ELECTRA), we get 
```
P_20                    all     0.4604
ndcg_cut_20             all     0.5399
```
That's good!
The above steps can also be done all at once by running `scripts/run.reranking.sh`.

### 3. Significance Test
To do a significance test, just configurate the `trec_eval` path in the `evaluation.py` file. 
Then simply run the following command, here we compare DocuBERT with BERT-MaxP:
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
DocuBERT achieve significant improvement over BERT-MaxP (p < 0.01) !