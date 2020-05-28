
import operator
import collections
import numpy as np
import tensorflow.compat.v1 as tf

class Relevance(object):

  def __init__(self, qid, judged_docno_list, supervised_docno_list, supervised_score_list):
    """

    :param qid: str, query id
    :param judged_docno_list: list of list, judged docnos from the result file
    :param supervised_docno_list: list, top K BM25 docnos
    :param supervised_score_list: list, top K BM25 document scores
    """

    self._qid = qid
    self._judged_docno_list = judged_docno_list
    self._supervised_docno_list = supervised_docno_list
    self._supervised_score_list = supervised_score_list

  def get_qid(self):
    return self._qid

  def get_judged_docno_list(self):
    return self._judged_docno_list

  def get_supervised_docno_list(self):
    return self._supervised_docno_list

  def get_supervised_score_list(self):
    return self._supervised_score_list


def update_qrels_from_res_and_qrels(qrels_file, res_dict):
  with open(qrels_file, 'r') as f:
    qrels = f.readlines()

  qrels_relevance_dict = collections.OrderedDict() # only from qrels
  prev_qid = ''

  relevance_map = collections.OrderedDict()
  # update from qrels
  for line in qrels:
    tokens = line.strip().split()
    qid, _, docno, relevance_score = tokens
    relevance_score, qid = int(relevance_score), str(qid)
    if qid != prev_qid:
      if len(relevance_map.values()) > 0:
        qrels_relevance_dict.update({prev_qid: relevance_map})
      prev_qid = qid
      relevance_map = collections.OrderedDict()
      relevance_map.update({docno: relevance_score})
    else:
      relevance_map.update({docno: relevance_score})
  qrels_relevance_dict.update({prev_qid: relevance_map})

  # update from result
  relevance_all = collections.OrderedDict()
  # we only consider qid occurs in both qrels and trec run file
  useful_qid = set(res_dict.keys()) & set(qrels_relevance_dict.keys())
  for qid in useful_qid:
    curr_supervised_docno_list, _ = res_dict.get(qid) # (docno_list, score_list)
    curr_qrels_map = qrels_relevance_dict.get(qid)
    relevance_posting = [[], [], []]
    for docno in curr_supervised_docno_list[:1000]: # from top 1000 result
      relevance_score = curr_qrels_map.get(docno)
      if relevance_score is None or relevance_score < 0:
        relevance_score = 0
      elif relevance_score > 2:
        relevance_score = 2
      else:
        pass # don't need to handle with labels 0, 1, 2, let it be
      relevance_posting[relevance_score].append(docno)
    if len(relevance_posting[1]) + len(relevance_posting[2]) < 5:
      tf.logging.warn("topic {0}: relevant document less than 5".format(qid))
    relevance_all.update({qid: relevance_posting})

  return relevance_all


def update_res_relevance(res_file):

  with open(res_file, 'r') as f:
    res = f.readlines()
  res_relevance_dict = collections.OrderedDict()
  prev_qid = ''
  docno_list, score_list = [], []

  for line in res:
    tokens = line.strip().split()
    qid, docno, rank, score = str(tokens[0]), tokens[2], int(tokens[3]), float(tokens[4])
    if qid != prev_qid:
      if len(docno_list) > 0:
        res_relevance_dict.update({prev_qid: (docno_list, score_list)})
      prev_qid = qid
      docno_list, score_list = [], []
      docno_list.append(docno)
      score_list.append(score)
    else:
      docno_list.append(docno)
      score_list.append(score)
  res_relevance_dict.update({prev_qid: (docno_list, score_list)})

  return res_relevance_dict


def create_relevance(result_filename, qrels_filename):
  relevance_dict = collections.OrderedDict()
  res_relevance_dict = update_res_relevance(result_filename)
  qid_list = res_relevance_dict.keys()
  qrels_relevance_dict = None
  if qrels_filename is not None:
    qrels_relevance_dict = update_qrels_from_res_and_qrels(qrels_filename, res_relevance_dict)
    # some qids don't have qrels, thus can be ignored
    qid_list = qrels_relevance_dict.keys()

  for qid in qid_list:
    supervised_docno_list, supervised_score_list = res_relevance_dict.get(qid)
    if qrels_filename is not None:
      judged_docno_list_within_supervised = qrels_relevance_dict.get(qid)
    else:
      judged_docno_list_within_supervised = None

    relevance = Relevance(qid, judged_docno_list_within_supervised, supervised_docno_list, supervised_score_list)
    relevance_dict.update({qid: relevance})

  return relevance_dict


def partition_fold(nb_fold, qrels_filename):
  ''' Based on the assumption that number of relevant docs in each fold
  should be as equal as possible.

  :param nb_fold:
  :param qrels_filename:
  :return:
  '''
  nb_positive_dict = collections.defaultdict(int)
  with open(qrels_filename, 'r') as f:
    for line in f:
      tokens = line.strip().split()
      qid, _, _, rel = tokens
      if int(rel) > 0:
        nb_positive_dict[qid] +=1

  nb_positive_tuple = sorted(nb_positive_dict.items(), key=operator.itemgetter(1))
  print(nb_positive_tuple)
  sorted_qid_list,  _ = zip(*nb_positive_tuple)
  partitions = [[] for i in range(nb_fold)]
  for i in range(len(sorted_qid_list) + nb_fold - 1 // nb_fold):
    candidate_qid = sorted_qid_list[i * nb_fold: (i + 1) * nb_fold]
    candidate_qid = np.random.permutation(candidate_qid)
    for j, qid in enumerate(candidate_qid):
      partitions[j].append(qid)
  total = sum([len(qids) for qids in partitions])
  assert total == len(nb_positive_dict.items())

  return partitions