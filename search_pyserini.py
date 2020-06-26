import re
import json
import pandas
import logging
import argparse
import collections
from bs4 import BeautifulSoup
from pyserini.search import SimpleSearcher
from utils.relevance_info import partition_fold

logging.basicConfig(level=logging.INFO)

def search(index_filename, query_filename, query_field, output_filename, runid, num_hits=1000):
  wf = open(output_filename, 'w')
  doc_wf = open(output_filename+"_rawdocs.txt", 'w')
  searcher = SimpleSearcher(index_filename)
  query_dict = read_query(query_filename, query_field)
  total_tokens = 0
  global_uniq_docids = set()
  for qid, query in query_dict.items():
    logging.info("Searching qid: {}, query: {}".format(qid, query))
    docids = set()
    num_duplicates = 0
    hits = searcher.search(query, num_hits)
    for rank, hit in enumerate(hits):

      docid = hit.docid
      score = hit.score
      if docid in docids:
        num_duplicates += 1
        continue
      docids.add(docid)
      # add new docs
      if docid not in global_uniq_docids:
        lucene_document = hit.lucene_document
        hit2json = json.loads(hit.raw)
        title = lucene_document.get('title')
        abstract = lucene_document.get('abstract')
        body = []
        if 'body_text' not in hit2json:
          body = ' '
        else:
          for bt in hit2json['body_text']:
            body.append(bt['text'])
          body = " ".join(body)
        regex = re.compile(r'[\n\r\t]')
        title = regex.sub(' ', title)
        abstract = regex.sub(' ', abstract)
        body = regex.sub(' ', body)
        write_to_text = " ".join([title, abstract, body])
        total_tokens += len(write_to_text.split())
        doc_wf.write("{}\t{}\n".format(docid, write_to_text))
        global_uniq_docids.add(docid)

      wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(qid, "Q0", docid, rank, score, runid))
    logging.warning("number of duplicate docs: {}".format(num_duplicates))
  wf.close()
  logging.info("Average document length: {}".format(total_tokens // len(global_uniq_docids)))



# raw = hits[0].raw
# hit2_json = json.loads(hits[0].raw)
# print(raw)
# print("=====================")
# print(json.dumps(hit2_json, indent=4))
#
# print(hits[0].lucene_document.getFields().stringValue())
# print("=====================")
# print(type(raw))
# print(raw.keys())



def read_query(topic_filename, field):

  with open(topic_filename, 'r') as f:
    topic = f.read()

  query_dict = collections.OrderedDict()
  soup = BeautifulSoup(topic, 'lxml')
  topics = soup.find_all('topic')
  for topic in topics:
    qid = topic['number']
    if '+' in field:
      fields = field.split("+")
      field_query = [topic.find_all(_field)[0].text.strip() for _field in fields]
      query = " ".join(field_query)
    else:
      query = topic.find_all(field)[0].text.strip()
    query_dict[qid] = query

  return query_dict

def read_meta_file(meta_filename):
  df = pandas.read_csv(meta_filename, sep=',', header=0)
  id_abstract_map = dict(zip(df.cord_uid, df.abstract))

  return id_abstract_map


def fetch_content_from_docid(index_filename, meta_filename, docid_filename, output_filename):
  searcher = pysearch.SimpleSearcher(index_filename)
  id_abstract_map = read_meta_file(meta_filename)
  total_tokens = 0
  with open(docid_filename, 'r') as rf, open(output_filename, 'w') as wf:
    for line in rf:
      docid = line.strip()
      doc = searcher.doc(docid)
      lucene_document = doc.lucene_document
      hit2json = json.loads(doc.raw())
      print(lucene_document)
      print("=========")
      print(hit2json)
      print("=========")
      print(id_abstract_map[docid])
      print("=========")
      assert 1==4
      title = lucene_document.get('title')
      abstract = lucene_document.get('abstract')
      body = []
      if 'body_text' not in hit2json:
        body = ' '
      else:
        for bt in hit2json['body_text']:
          body.append(bt['text'])
        body = " ".join(body)
      regex = re.compile(r'[\n\r\t]')
      title = regex.sub(' ', title)
      abstract = regex.sub(' ', abstract)
      body = regex.sub(' ', body)
      write_to_text = " ".join([title, abstract, body])
      total_tokens += len(write_to_text.split())
      doc_wf.write("{}\t{}\n".format(docid, write_to_text))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("index_filename", help="index path")
  parser.add_argument("meta_filename", help="covid meta filename")
  parser.add_argument("docid_filename", help="docid path, one docid one line")
  parser.add_argument("output_filename", help="output path, format: docid \t content")
  args = parser.parse_args()
  fetch_content_from_docid(
    args.index_filename,
    args.meta_filename,
    args.docid_filename,
    args.output_filename)
  # hits = 10000
  # runid = "udel_fang"
  # field = 'query'
  # index_filename = "/data2/index/lucene-index-cord19-full-text-2020-05-19/"
  # # index_filename = "/data2/index/lucene-index-cord19-full-text-2020-05-01/"
  # # index_filename = "/data2/index/lucene-index-covid-full-text-2020-04-10"
  # # query_filename = "/data/tool/anserini/src/main/resources/topics-and-qrels/topics.covid-round2.xml"
  # query_filename = "/data/tool/anserini/src/main/resources/topics-and-qrels/topics.covid-round3-udel.xml"
  #
  # output_filename = "/data2/covid19/udel_fang/runs/round3.bm25.2020-05-19.recall-{}.{}.txt".format(hits, runid)
  # # output_filename = "/data2/covid19/udel_fang/runs/round2.bm25.2020-05-01.recall-10000.txt"
  # # runid="query_question"
  #
  # # avg length = 5850
  #
  # # search(index_filename, query_filename, field, output_filename, runid, num_hits=hits)
  # l = partition_fold(5, "/data/tool/anserini/src/main/resources/topics-and-qrels/qrels.covid-round12.txt")
  # print(l)