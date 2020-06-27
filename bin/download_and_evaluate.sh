gs_dir=$1
local_dir=$2
qrels_path=$3
epoch=$4
filter_qrels_path=$5

mkdir -p $local_dir
rm ${local_dir}/*

gsutil cp $gs_dir/fold_*epoch_${epoch}_bert_predictions_test.txt $local_dir
# there is supposed to be only one file
filename=$(ls ${local_dir})

python3 sanity_check.py \
  --run ${local_dir}/${filename} \
  --qrels ${filter_qrels_path}


trec_eval ${qrels_path} ${local_dir}/${filename}".filter" -m map -m recall.1000 -m P.5 -m ndcg_cut.10 -m bpref >> ${local_dir}/result_epoch${epoch}
cat ${local_dir}/result_epoch${epoch}

