import gzip
import csv
import fire

# In the corpus tsv, each docid occurs at offset docoffset[docid]
def get_offset(offset_filename="msmarco-docs-lookup.tsv.gz"):
  doc_offset = {}
  with gzip.open(offset_filename, 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
      doc_offset[docid] = int(offset)

  return doc_offset


def get_corpus(docid_filename, corpus_filename, offset_filename, output_filename):
  doc_offset = get_offset(offset_filename)
  with open(docid_filename, 'r') as id_f, \
      open(corpus_filename, 'r') as corpus_f, \
      open(output_filename, 'w') as out_f:
    for idx, line in enumerate(id_f):
      if idx % 1000 == 0:
        print("Fetching {} lines".format(idx))
      docid = line.strip()
      corpus_f.seek(doc_offset[docid])
      doc_info = corpus_f.readline()
      assert line.startswith(docid + "\t"), 'Looking for {docid}, found {line}'.format(docid=docid, line=line)
      _, _, title, body = doc_info.split("\t")
      out_f.write("{}\t{}".format(docid, " ".join([title, body])))

def main():
  fire.Fire(get_corpus)

if __name__ == '__main__':
  main()