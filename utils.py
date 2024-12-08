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
    stop_en = stopwords.words('english')
    analyzer = StandardAnalyzer(stoplist=stop_en) 
    tokenVector = [token.text for token in analyzer(text)]
    stemmer = PorterStemmer()
    processedVector = [stemmer.stem(token) for token in tokenVector]

    return processedVector

def extractSubQueries(tokenVector ,NGRAM=1):
    if(NGRAM > 1):
        queryVector = []
        tokenVector = [ list(tulp) for tulp in ngrams(tokenVector, NGRAM)]
        for token in tokenVector:
            query = whoosh.query.Phrase("content", token)
            queryVector.append(query)
        return queryVector

    return tokenVector

def expandQuery(queries):
    pass


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

    # print([term for term in ix.reader().all_terms()])
    return (index_process_time, index_time)


def searchDoc(query):
    ix = index.open_dir("index")

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        # Search
        results = searcher.search(query, scored=True, limit=10)

        print([(item.fields()['reference'], item.score) for item in results])

        # new_terms = results.key_terms("content", numterms=10, docs=5)
        # print("terms")

        # expanded_query = whoosh.query.Or([whoosh.query.Term("content", word, boost=weight) for word, weight in new_terms])
        # print("expanded_query")

        # expanded_results =  searcher.search(expanded_query, scored=True, limit=10)
        # print("expanded_results")

        # print(expanded_results)

        # print([(item.fields()['reference'], item.score) for item in expanded_results])
        # print("\n")

        # results.upgrade_and_extend(expanded_results)

        processed_result = [(result.fields(), result.score) for result in results]

        # processed_result.sort(key=lambda result: result[1], reverse=True)
    

        return processed_result