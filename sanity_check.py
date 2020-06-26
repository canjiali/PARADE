
import argparse
import collections
from utils import relevance_info


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--run1", help="first run")
  parser.add_argument("--run2", help="second run")
  parser.add_argument("--run", help="run path")
  parser.add_argument("--qrels", help="qrels path")
  args = parser.parse_args()

  # rel1 = relevance_info.create_relevance(args.run1, None)
  # rel2 = relevance_info.create_relevance(args.run2, None)
  # assert len(rel1.keys()) == len(rel2.keys()), "number of qids differs"
  # overlap = 0
  # for qid in rel1.keys():
  #   sup1 = rel1[qid].get_supervised_docno_list()[:100]
  #   sup2 = rel2[qid].get_supervised_docno_list()[:100]
  #   overlap += len(set(sup1).intersection(set(sup2)))
  # print(overlap)
  # print(overlap/len(rel1.keys()))
  #
  # for qid in rel1.keys():
  #   sup1 = rel1[qid].get_supervised_docno_list()
  #   sup2 = rel2[qid].get_supervised_docno_list()
  #   assert sorted(sup1) == sorted(sup2), "qid {} does not pass the test".format(qid)
  #
  # print("Pass Test!")

  # The filtering stuff
  existing_pairs = set()
  with open(args.qrels, 'r') as f:
    for line in f:
      tokens = line.strip().split()
      qid, _, docid, _ = tokens
      existing_pairs.add("\t".join([qid, docid]))

  count_dict = collections.defaultdict(int)
  with open(args.run, 'r') as rf, open(args.run+'.filter', 'w') as wf:
    count = 0
    for line in rf:
      tokens = line.strip().split()
      qid, _, docid, _, _, _ = tokens
      if count_dict[qid] >= 1000:
        count += 1
        continue
      if "\t".join([qid, docid]) not in existing_pairs:
        wf.write(line)
        count_dict[qid] += 1
      else:
        count += 1
  print("Filter {} lines".format(count))
