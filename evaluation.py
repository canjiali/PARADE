
import os
import re
import sys
import argparse
import subprocess

from scipy import stats
from collections import OrderedDict


parent_path = '/data/tool'
trec_eval_script_path = os.path.join(parent_path, 'trec_eval-9.0.7/trec_eval')
sample_eval_script_path = os.path.join(parent_path, "sample_eval.pl")
gd_eval_script_path = os.path.join(parent_path, "gdeval.pl")


def run(command, get_ouput=False):
  try:
    if get_ouput:
      process = subprocess.Popen(command, stdout=subprocess.PIPE)
      output, err = process.communicate()
      exit_code = process.wait()
      return output
    else:
      subprocess.call(command)
  except subprocess.CalledProcessError as e:
    print(e)


def evaluate_trec(qrels, res, metrics):

  ''' all_trecs, '''
  command = [trec_eval_script_path, '-m', 'all_trec', '-M', '1000', qrels, res]
  output = run(command, get_ouput=True)

  metrics_val = []
  for metric in metrics:
    metrics_val.append(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[2].strip())

  # MAP = re.findall(r'map\s+all.+\d+', output)[0].split('\t')[2].strip()
  # P20 = re.findall(r'P_20\s+all.+\d+', output)[0].split('\t')[2].strip()

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_sample_trec(qrels, res, metrics):
  command = [sample_eval_script_path, qrels, res]
  output = run(command, get_ouput=True)

  metrics_val = []
  for metric in metrics:
    metrics_val.append(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[4].strip())

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_metrics(qrels, res, sample_qrels=None, metrics=None):
  normal_metrics = [met for met in metrics if not met.startswith('i')]
  infer_metrics = [met for met in metrics if met.startswith('i')]

  metrics_val_dict = OrderedDict()
  if len(normal_metrics) > 0:
    metrics_val_dict.update(evaluate_trec(qrels, res, metrics=normal_metrics))
  if len(infer_metrics) > 0:
    metrics_val_dict.update(evaluate_sample_trec(sample_qrels, res, metrics=infer_metrics))

  return metrics_val_dict

################################## perquery information ####################################
def evaluate_trec_perquery(qrels, res, metrics):

  ''' all_trecs, '''
  command = [trec_eval_script_path, '-m', 'all_trec', '-q', '-M', '1000', qrels, res]
  output = run(command, get_ouput=True)

  metrics_val = []
  for metric in metrics:
    curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
    curr_res = map(lambda x: float(x.split('\t')[-1]), curr_res)
    metrics_val.append(curr_res)

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_sample_trec_perquery(qrels, res, metrics):
  command = [sample_eval_script_path, '-q', qrels, res]
  output = run(command, get_ouput=True)

  metrics_val = []
  for metric in metrics:
    curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
    curr_res = map(lambda x: float(x.split('\t')[-1]), curr_res)
    metrics_val.append(curr_res)

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_metrics_perquery(qrels, res, sample_qrels=None, metrics=None):
  normal_metrics = [met for met in metrics if not met.startswith('i')]
  infer_metrics = [met for met in metrics if met.startswith('i')]

  metrics_val_dict = OrderedDict()
  if len(normal_metrics) > 0:
    metrics_val_dict.update(evaluate_trec_perquery(qrels, res, metrics=normal_metrics))
  if len(infer_metrics) > 0:
    metrics_val_dict.update(evaluate_sample_trec_perquery(sample_qrels, res, metrics=infer_metrics))

  return metrics_val_dict


def tt_test(qrels, res1, res2, sample_qrels=None, metrics=None):
  met_dict1 = evaluate_metrics_perquery(qrels, res1, sample_qrels, metrics)
  met_dict2 = evaluate_metrics_perquery(qrels, res2, sample_qrels, metrics)

  avg_met_dict1 = evaluate_metrics(qrels, res1, sample_qrels, metrics)
  avg_met_dict2 = evaluate_metrics(qrels, res2, sample_qrels, metrics)
  print(avg_met_dict1)
  print(avg_met_dict2)

  test_dict = OrderedDict()
  for met in met_dict1.keys():
    p_value = stats.ttest_rel(met_dict1.get(met), met_dict2.get(met))[1]
    test_dict.update({met: p_value})
  print(test_dict)

  return test_dict

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--qrels", help="TREC qrels file path")
  parser.add_argument("--baselines", help="Baseline file paths, seperated by ','")
  parser.add_argument("--runs", help="competitive run paths, seperated by ','")
  args = parser.parse_args()

  baselines = args.baselines.split(",")
  runs = args.runs.split(",")


  for trec_run in runs:
    for baseline in baselines:
      print(baseline)
      print(trec_run)
      tt_test(args.qrels, baseline, trec_run, None, ['P_20', 'ndcg_cut_20'])
