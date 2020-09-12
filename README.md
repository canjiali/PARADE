# PARADE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3974431.svg)](https://doi.org/10.5281/zenodo.3974431)

This repository contains the code for our paper
- PARADE: Passage Representation Aggregation for Document Reranking

If you're interested in running PARADE for the TREC-COVID challenge (submitted with tag `mpiid5` from Round 2), 
please check out the **covid** branch.

If you find this paper/code useful, please cite:
```
@article{li2020parade,
  title={PARADE: Passage Representation Aggregation for Document Reranking},
  author={Li, Canjia and Yates, Andrew and MacAvaney, Sean and He, Ben and Sun, Yingfei},
  journal={arXiv preprint arXiv:2008.09093},
  year={2020}
}
```

## Introduction
PARADE (PAssage Representation Aggregation for Document rE-ranking) is an end-to-end document reranking model based on the pre-trained language models.

We support the following PARADE variants:
- PARADE-Avg (named `cls_avg` in the code)
- PARADE-Attn (named `cls_attn` in the code)
- PARADE-Max (named `cls_max` in the code)
- PARADE (named `cls_transformer` in the code)

We support two instantiations of pre-trained models:
- BERT
- ELECTRA

## Getting Started
To run PARADE, there're two steps ahead.
We give a detailed example on how to run the code the Robust04 dataset using the title query.

### 1. Data Preparation
To run a 5-fold cross-validation, data for 5 folds are required.
The standard `qrels`, `query`, `trec_run` files can be accomplished by [Anserini](https://github.com/castorini/anserini),
please check out their notebook for further details.
Then you need to split the documets into passages, write them into TFrecord files.
The `corpus` file can also be extracted by Anserini to form a `docno \t content` paired text.
Then run

```bash
scripts/run.convert.data.sh
```
You should be able to see 5 sub-folders generated in the `output_dir` folder,
with each contains a train file and a test file.
Note that if you're going to run the code on TPU, you need to upload the training/testing data to Google Cloud Storage (GCS).
Everything is prepared now!

### 2. Model Traning and Evaluation


For all the pre-trained models, we first fine-tune them on the MSMARCO passage collection.
This is IMPORTANT, as it can improve the nDCG@20 by 2 points generally.
To figure out the way of doing that, please check out [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert).
If you want to escape this fine-tuning step,
check out these fine-tuned [models](#resource) on the MSMARCO passage ranking dataset.
The fine-tuned model will be the initialized model in PARADE.
Just pass it to the `BERT_ckpt` argument in the following snippet. 

Now train the model:

```bash
scripts/run.reranking.sh
```

The model performance will automatically output on your screen. 
When evaluating the title queries on the Robust04 collecting, it outputs
```
P_20                    all     0.4604
ndcg_cut_20             all     0.5399
```



# <a name="resource"></a> Useful Resources

- Fine-tuned models on the MSMARCO passage ranking dataset:

| Model        | L / H    | MRR on MSMARCO DEV | Path |
|--------------|----------|--------------------|------|
| ELECTRA-Base | 12 / 768 | 0.3698     | [Download](https://zenodo.org/record/3974431/files/vanilla_electra_base_on_MSMARCO.tar.gz)    |
| BERT-Base    | 12 / 768 | 0.3637     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_base_on_MSMARCO.tar.gz)    |
| \            | 10 / 768 | 0.3622     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_medium_10_base_on_MSMARCO.tar.gz)    |
| \            | 8 / 768  | 0.3560     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_medium_8_base_on_MSMARCO.tar.gz)   |
| BERT-Medium  | 8 / 512  | 0.3520     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_medium_on_MSMARCO.tar.gz)    |
| BERT-Small   | 4 / 512  | 0.3427     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_mini_on_MSMARCO.tar.gz)   |
| BERT-Mini    | 4 / 256  | 0.3247     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_mini_on_MSMARCO.tar.gz)  |
| \            | 2 / 512  | 0.3160     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_tiny_small_on_MSMARCO.tar.gz)    |
| BERT-Tiny    | 2 /128   | 0.2600     | [Download](https://zenodo.org/record/3974431/files/vanilla_bert_tiny_on_MSMARCO.tar.gz)   |

- Our run files on the Robust04 and GOV2 collections: 
[Robust04](https://zenodo.org/record/3974431/files/robust04.PARADE.runs.tar.gz), 
[GOV2](https://zenodo.org/record/3974431/files/gov2.PARADE.runs.tar.gz).

# FAQ
- How to get the raw text?

If you bother getting the raw text from Anserini, 
you can also replace the `anserini/src/main/java/io/anserini/index/IndexUtils.java` file by the `extra/IndexUtils.java` file in this repo,
then re-build Anserini (version 0.7.0).
Below is how we fetch the raw text
```bash
anserini_path="path_to_anserini"
index_path="path_to_index"
# say you're given a BM25 run file run.BM25.txt
cut -d ' ' -f3 run.BM25.txt | sort | uniq > docnolist
${anserini_path}/target/appassembler/bin/IndexUtils -dumpTransformedDocBatch docnolist -index ${index_path}
```
then you get the required raw text in the directory that contains docnolist. 
Alternatively, you can refer to the `search_pyserini.py` file in the **covid** branch and fetch the docs using pyserini.

- How to run a significance test?

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
The last line shows the p-values.
PARADE achieves significant improvement over BERT-MaxP (p < 0.01) !

- How to run knowledge distillation for PARADE?

Please follow the fine-tuning steps first.
Then run the following command:
```bash
scripts/run.kd.sh
```
It outputs the following results with regard to PARADE using the BERT-small model (4 layers)!
```bash
P_20                    all     0.4365
ndcg_cut_20             all     0.5098
```

# Acknowledgement
Some snippets of the codes are borrowed from 
[NPRF](https://github.com/ucasir/NPRF),
[Capreolus](https://github.com/capreolus-ir/capreolus),
[dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert),
[SIGIR19-BERT-IR](https://github.com/AdeDZY/SIGIR19-BERT-IR).
