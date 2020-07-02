
import argparse
import collections
import numpy as np

from utils import relevance_info
from utils import result_info


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--shallow_run", help="first run")
  parser.add_argument("--deep_run", help="second run")
  parser.add_argument("--shallow_threshold", type=int, help="the reranking threshold")
  parser.add_argument("--deep_threshold", type=int, help="the reranking threshold")
  parser.add_argument("--merge_run", help="where to store the run")
  args = parser.parse_args()

  shallow_rel = relevance_info.create_relevance(args.shallow_run)
  deep_rel = relevance_info.create_relevance(args.deep_run)
  result_dict = collections.OrderedDict()
  for qid in shallow_rel.keys():
    curr_shallow_rel = shallow_rel.get(qid)
    curr_deep_rel = deep_rel.get(qid)
    shallow_docs = curr_shallow_rel.get_supervised_docno_list()[:args.shallow_threshold]
    shallow_scores = curr_shallow_rel.get_supervised_score_list()[:args.shallow_threshold]
    deep_docs = curr_deep_rel.get_supervised_docno_list()[:args.deep_threshold]
    deep_scores = curr_deep_rel.get_supervised_score_list()[:args.deep_threshold]
    # filter
    deep_docs = [doc for doc in deep_docs if doc not in shallow_docs]
    behind_score = np.min(shallow_scores) - 0.001 - np.sort(
      np.random.random(len(deep_docs)))
    doc_list = shallow_docs + deep_docs
    score_list = np.concatenate((shallow_scores, behind_score))

    res = result_info.Result(qid, doc_list, score_list, "merge_run")
    res.update_ranking()
    result_dict.update({qid: res})
  result_info.write_result_to_trec_format(result_dict, args.merge_run)
