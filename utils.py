import nltk
nltk.download('stopwords')
import os, glob
from utils import *
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

def folderReader(path, encoding='utf-8'):
    os.chdir(path)
    files = []
    for file in glob.glob("*/*.txt"):
        files.append(file) 
    return files

def zipfGrapgh(path, encoding='utf-8'):
    fd = FreqDist()
    files = folderReader(path)

    for file in files:
        with open(file, 'r', encoding=encoding) as f:
            text = f.read()
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