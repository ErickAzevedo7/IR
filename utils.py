import nltk
import whoosh
import xml.etree.ElementTree as ET
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import Schema, STORED, TEXT, ID
import whoosh.fields
import whoosh.index as index
import os, glob

import whoosh.reading
from utils import *
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt

def folderReader(path):
    files = []
    for file in glob.glob(os.path.join(path, "*/*.txt")):
        base = os.path.splitext(file)[0]
        xmlfile = os.path.join(base + ".xml")
        document = [file, xmlfile]

        files.append(document)
        break
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

    return tokenVector


def indexDoc(files):
    schema = Schema(title=STORED, reference=ID(stored=True), content=TEXT(vector=True))

    if not os.path.exists("index"):
        os.mkdir("index")

    ix = index.create_in("index", schema)

    writer = ix.writer()

    for file in files:
        rawText = fileReader(file[0])
        text = preProcess(rawText)

        tree = ET.parse(file[1]).getroot()
        
        reference = tree.get('reference')

        title = tree.find('feature').get('title')
        
        writer.add_document(title=title, reference=reference, content=text)

    writer.commit()