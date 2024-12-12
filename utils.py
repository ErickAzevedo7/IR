import whoosh
import xml.etree.ElementTree as ET
from whoosh.analysis import StandardAnalyzer
import whoosh.analysis
import whoosh.analysis.filters
from whoosh.fields import Schema, STORED, TEXT, ID, NGRAMWORDS, KEYWORD
from whoosh import scoring
import whoosh.fields
import whoosh.index as index
import os, glob
import time
import re
import whoosh.qparser
import whoosh.query
import whoosh.reading
import whoosh.searching
from utils import *
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
import string
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

client = Elasticsearch(os.getenv('ES_LOCAL_URL'), api_key=(os.getenv('ES_LOCAL_API_KEY')), timeout=400)

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def folderReader(path):
    files = []
    for file in glob.glob(os.path.join(path, "*/*.txt")):
        base = os.path.splitext(file)[0]
        xmlfile = os.path.join(base + ".xml")
        document = [file, xmlfile]

        files.append(document)
    return files

def fileReader(file, encoding='utf-8'):
    with open(file, 'r', encoding=encoding) as f:
        text = f.read()
    return text

def zipfGrapgh(path, encoding='utf-8'):
    fd = FreqDist()
    files = folderReader(path)

    for file in files:
        text = fileReader(file[0], encoding)
        text = text.translate(str.maketrans('','',string.punctuation))
        fd += FreqDist(word.lower() for word in text.split())

    print("Frequencies done!")
    print("Tamanho do vocabulário:", fd.B())
    print("Número total de palavras:", fd.N())
    print("10 palavras mais frequentes da colecao:", fd.pformat(10))
    print("10 palavras menos frequentes da colecao:", fd.most_common()[-10:])
    print("stop words:", list(stopwords.words('english')))
    p = fd.plot(10,show=False,title="Distribuição das palavras na coleção de plagio")
    p.set_xlabel("Amostra")
    p.set_ylabel("Frequência")
    plt.show()


def preProcess(text):
    text_nonum = re.sub(r'\d+', '', text)
    stop_en = stopwords.words('english')
    analyzer = StandardAnalyzer(stoplist=stop_en) 
    tokenVector = [token.text for token in analyzer(text_nonum)]
    stemmer = PorterStemmer()
    processedVector = [stemmer.stem(token) for token in tokenVector]

    return processedVector

def extractSubQueries(tokenVector ,NGRAM=1):
    if(NGRAM > 1):
        queryVector = []
        tokenVector = [list(tulp) for tulp in ngrams(tokenVector, NGRAM)]
        for token in tokenVector:
            query = whoosh.query.Phrase("content", token)
            queryVector.append(query)
        return queryVector
    else:
        queryVector = []
        for token in tokenVector:
            query = whoosh.query.Term("content", token)
            queryVector.append(query)

    return queryVector

def extractSubQueriesElastic(tokenVector, NGRAM=1):
 
    queryVector = []
    tokenVector = [list(tulp) for tulp in ngrams(tokenVector, NGRAM)]
    for token in tokenVector:
        query = {
            "match_phrase": {
                "content": " ".join(token)
            }
        }
        queryVector.append(query)

    return queryVector

def expandQuery(queries):
    pass

def indexDocElastic(files):
    stop_en = stopwords.words('english')
    if client.indices.exists(index="elastic_index"):
        client.indices.delete(index="elastic_index")

    resp = client.indices.create(
        index="elastic_index",
        settings={
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "my_stop_words",
                            "lowercase",
                            "porter_stem"
                            ]
                    }
                },
                "filter": {
                    "my_stop_words": {
                        "type": "stop",
                        "stopwords": stop_en
                    }
                }
            }
        },
        mappings={
            "properties": {
                "reference": {
                    "type": "keyword",
                },
                "title": {
                    "type": "text",
                },
                "content": {
                    "type": "text",
                    "analyzer": "default",
                }
            }
        },
    )

    index_process_time = []
    index_time = []
    count = 0
    for file in files:
        rawText = fileReader(file[0])

        start = time.time()
        tree = ET.parse(file[1]).getroot()
        
        reference = tree.get('reference')

        title = tree.find('feature').get('title')
        
        client.index(index="elastic_index", document={"title": title, "reference": reference, "content": rawText})
        index_time.append(time.time() - start)

        client.indices.refresh(index="elastic_index")


    return (index_process_time, index_time)

def indexDoc(files):
    schema = Schema(title=STORED, reference=ID(stored=True), content=TEXT(vector=True))

    if not os.path.exists("index"):
        os.mkdir("index")

    ix = index.create_in("index", schema)

    writer = ix.writer()

    index_process_time = []
    index_time = []
    for file in files:
        rawText = fileReader(file[0])

        start = time.time()
        text = preProcess(rawText)
        index_process_time.append(time.time() - start)

        tree = ET.parse(file[1]).getroot()
        
        reference = tree.get('reference')

        title = tree.find('feature').get('title')
        
        writer.add_document(title=title, reference=reference, content=text)
        index_time.append(time.time() - start)

    writer.commit()

    return (index_process_time, index_time)

def searchDocElastic(queries):
    result = client.search(
        index="elastic_index",
        query={
            "dis_max": {
                "queries": queries
            },
        }
    )

    return result.body['hits']['hits']

    

def searchDoc(query):
    ix = index.open_dir("index")

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        query.normalize()

        results = searcher.search(query, scored=True, limit=10)

        new_terms = results.key_terms("content", numterms=10, docs=5)

        expanded_query = whoosh.query.Or([whoosh.query.Term("content", word, boost=weight) for word, weight in new_terms])

        expanded_results =  searcher.search(expanded_query, scored=True, limit=10)

        results.upgrade_and_extend(expanded_results)

        processed_result = [(result.fields(), result.score) for result in results]

        processed_result.sort(key=lambda result: result[1], reverse=True)
    

        return processed_result