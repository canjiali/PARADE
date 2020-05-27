
import numpy as np
import tensorflow.compat.v1 as tf

import collections
from utils import file_operation

class Result(object):

  def __init__(self, qid, docno_list, score_list, runid):
    self._qid = qid
    self._docno_list = docno_list
    self._score_list = score_list
    self._runid = runid

  def get_qid(self):
    return self._qid

  def get_docno_list(self):
    return self._docno_list

  def get_score_list(self):
    return self._score_list

  def get_runid(self):
    return self._runid

  def update_ranking(self):
    pair = zip(self._docno_list, self._score_list)
    updated_pair = sorted(pair, key=lambda x: x[1], reverse=True)
    self._docno_list, self._score_list = zip(*updated_pair)

  def set_docno_list(self, docno_list):
    self._docno_list = docno_list

  def set_score_list(self, score_list):
    self._score_list = score_list

  def update_all(self, docno_list, score_list):
    self.set_docno_list(docno_list)
    self.set_score_list(score_list)
    self.update_ranking()


def write_result_to_trec_format(result_dict, write_path):

  f = tf.gfile.Open(write_path, 'w')
  for qid, result in result_dict.items():

    docno_list = result.get_docno_list()
    score_list = result.get_score_list()
    rank = 0
    for docno, score in zip(docno_list, score_list):
      f.write("{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(qid, docno, rank, score, result.get_runid()))
      rank += 1

  f.close()


def read_result_from_file(result_file):

  res_all = file_operation.parse_corpus(result_file)
  res_dict = collections.OrderedDict()
  prev_qid, runid = -1, -1
  docno_list = []
  score_list = []
  for line in res_all:
    tokens = line.split()
    qid, docno, score, runid = int(tokens[0]), tokens[2], float(tokens[4]), tokens[5]
    if qid != prev_qid:
      if len(docno_list) > 0:
        result = Result(qid, docno_list, score_list, runid)
        res_dict.update({prev_qid: result})
      docno_list, score_list = [docno], [score]
      prev_qid = qid
    else:
      docno_list.append(docno)
      score_list.append(score)
  result = Result(prev_qid, docno_list, score_list, runid)
  res_dict.update({prev_qid: result})

  return res_dict


def write_result_from_score(rerank_topk, scores, qid_list, relevance_dict, write_path, runid):
  res_dict = collections.OrderedDict()
  num_of_reranking_should_be = 0
  accumulator = 0
  for i, qid in enumerate(qid_list):
    relevance = relevance_dict.get(qid)
    supervised_docno_list = relevance.get_supervised_docno_list()
    num_rerank = min(rerank_topk, len(supervised_docno_list[: rerank_topk]))
    topk_score = scores[accumulator: accumulator + num_rerank]
    accumulator += num_rerank
    num_of_reranking_should_be += num_rerank

    # trec_eval's ranking is based on the score.
    if len(supervised_docno_list) <= rerank_topk:
      score_list = topk_score
    else:
      behind_score = np.min(topk_score) - 0.001 - np.sort(
        np.random.random((len(supervised_docno_list) - rerank_topk,)))
      score_list = np.concatenate((topk_score, behind_score))

    res = Result(qid, supervised_docno_list, score_list, runid)
    res.update_ranking()
    res_dict.update({qid: res})
  write_result_to_trec_format(res_dict, write_path)
  tf.logging.info("Number of docs should be re-ranked {}".format(num_of_reranking_should_be))