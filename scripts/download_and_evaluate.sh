
gs_dir=$1
local_dir=$2
qrels_path=$3
epoch=$4

mkdir -p $local_dir

for fold in {1..5}
do
  gsutil cp $gs_dir/fold-${fold}/fold_*epoch_${epoch}_bert_predictions_test.txt $local_dir
done

# make sure that we find out all the 5 result files
# if you're in a local machine, you don't need to download
num_result=$(ls $local_dir |wc -l)
if [ "$num_result" != "5" ]; then
  echo Exit. Wrong number of results, only $num_result result files found!
  exit
fi

cat ${local_dir}/fold_*epoch_${epoch}_bert_predictions_test.txt >> ${local_dir}/merge_epoch${epoch}
/data/tool/trec_eval-9.0.7/trec_eval ${qrels_path} ${local_dir}/merge_epoch${epoch} -m ndcg_cut.20 -m P.20 -m map >> ${local_dir}/result_epoch${epoch}
cat ${local_dir}/result_epoch${epoch}
