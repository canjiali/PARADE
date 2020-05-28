# -*- coding: UTF-8 -*-

import gzip
import collections

from collections import defaultdict
from bs4 import BeautifulSoup


def load_ntcir_query(topic_filename, fields):
  with open(topic_filename, 'r') as f:
    topic = f.read()
  query_dict = collections.OrderedDict()
  soup = BeautifulSoup(topic, 'lxml')
  topics = soup.find_all('query')
  if '+' in fields:
    fields = fields.split("+")
  else:
    fields = [fields]
  for topic in topics:
    qid = topic.find_all("qid")[0].text.strip()
    query = []
    for _field in fields:
      query.append(topic.find_all(_field)[0].text.strip())
    query = " ".join(query)
    query_dict[qid] = query
  return query_dict


def load_xml_query(topic_filename, field):

  with open(topic_filename, 'r') as f:
    topic = f.read()

  query_dict = collections.OrderedDict()
  soup = BeautifulSoup(topic, 'lxml')
  topics = soup.find_all('topic')
  for topic in topics:
    qid = str(topic['number'])
    query = topic.find_all(field)[0].text.strip()
    query_dict[qid] = query

  return query_dict


def load_trec_topics(queryfn):
  """  Modified from https://github.com/capreolus-ir/capreolus/blob/413b3bacc5cb9afd6c36e465a804d30456cb31a4/capreolus/utils/trec.py

  :param queryfn:
  :return:
  """
  title, desc, narr = defaultdict(list), defaultdict(list), defaultdict(list)
  block = None
  if queryfn.endswith(".gz"):
    openf = gzip.open
  else:
    openf = open
  with openf(queryfn, "rt") as f:
    for line in f:
      line = line.strip()
      if line.startswith("<num>"):
        # <num> Number: 700
        qid = line.split()[-1]
        qid = str(qid)
        # no longer an int
        # assert qid > 0
        block = None
      elif line.startswith("<title>"):
        # <title>  query here
        title[qid].extend(line.strip().split()[1:])
        block = "title"
        # TODO does this sometimes start with Topic: ?
        assert "Topic:" not in line
      elif line.startswith("<desc>"):
        # <desc> description \n description
        desc[qid].extend(line.strip().split()[1:])
        block = "desc"
      elif line.startswith("<narr>"):
        # same format as <desc>
        narr[qid].extend(line.strip().split()[1:])
        block = "narr"
      elif line.startswith("</top>") or line.startswith("<top>"):
        block = None
      elif block == "title":
        title[qid].extend(line.strip().split())
      elif block == "desc":
        desc[qid].extend(line.strip().split())
      elif block == "narr":
        narr[qid].extend(line.strip().split())
  out = {}
  if len(title) > 0:
    out["title"] = {qid: " ".join(terms) for qid, terms in title.items()}
  if len(desc) > 0:
    out["desc"] = {qid: " ".join(terms) for qid, terms in desc.items()}
  if len(narr) > 0:
    out["narr"] = {qid: " ".join(terms) for qid, terms in narr.items()}

  # starts with "Description:" may be annoying
  for qid, query in out["desc"].items():
    if query.startswith("Description:"):
      query = query[12:].strip()
    out["desc"][qid] = query

  return out


def key_value_from_file(filename, convert_to_int=False):
  result_dict = collections.OrderedDict()
  with open(filename, 'r') as f:
    for line in f:
      segments = line.strip().split("\t")
      if len(segments) == 2:
        docno, content = segments
      elif len(segments) < 2:
        # only docid
        docno = line.strip()
        content = "It is empty."
      else:
        # multiple '\t' occur
        docno = segments[0]
        content = " ".join(segments[1:])
      if convert_to_int:
        docno = int(docno)
      result_dict.update({docno: content})
  return result_dict


def key_value_from_trec_DL_file(filename):
  result_dict = collections.OrderedDict()
  with open(filename, 'r') as f:
    for line in f:
      try:
        segments = line.strip().split("\t")
        docno, url, title, content = segments
      except ValueError:
        if len(segments) == 3:
          docno, url, title = segments
          content = "It is empty"
          if title == '.':
            title = "It is empty title"
        elif len(segments) == 2:
          docno, url = segments
          title = "It is empty title"
          content = "It is empty"
        else:
          docno = line.strip()
          title = "It is empty title"
          content = "It is empty"
        # some lines may not have two values.
      result_dict.update({docno: (title, content)})
  return result_dict


def parse_corpus(corpus_file):
  corpus = []
  with open(corpus_file, 'r') as f:
    for line in f:
      corpus.append(line.strip())

  return corpus

