from utils import *
import time
import numpy as np
import xml.etree.ElementTree as ET
import json
source = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/source-document'
suspect = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document'
json_path = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/papers.json'

if __name__ == '__main__':
    # files = folderReader(source)

    # index_process_time, index_time = indexDoc(files)

    # print("Process index: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(index_process_time),sum(index_process_time)/len(index_process_time), len(index_process_time)))
    # print("Process index: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(index_time),sum(index_time)/len(index_time), len(index_time)))

    with open(json_path, 'r') as f:
        data = json.load(f)

    total_precision = [0, 0, 0, 0, 0]
    total_recall = [0, 0, 0, 0, 0]
    query_process_time = []
    query_time = []
    
    for i in range(64):
        source = find(data[i]['filename'], suspect)

        text = fileReader(source)

        # Preprocess text
        start_preprocess = time.time()
        p_text = preProcess(text)
        query_process_time.append(time.time() - start_preprocess)

        print("tempo: %f" % query_process_time[0])

        # Extract subqueries
        subqueries = extractSubQueries(p_text, 5)

        query = whoosh.query.DisjunctionMax(subqueries)

        results = searchDoc(query)
        query_time.append(time.time() - start_preprocess)

        # for result in results:
        #     print(result[0]['reference'], result[1])

        pAtK = []
        rAtk = []
        scalex = []
        hits = 0
        for i in range(len(results[:10])):  
            for doc in data[2]['src_file']:
                if(results[i][0]['reference'] == doc):
                    # print(results[i][0]['reference'], results[i][1])
                    hits += 1
            if i == 1 or i == 3 or i == 5 or i == 7 or i == 9:
                pAtK.append(hits/(i+1)) 
                rAtk.append(hits/len(data[2]['src_file']))
                scalex.append(i+1)

        for i in range(len(pAtK)):
            total_precision[i] = (total_precision[i] + pAtK[i])
        
        for i in range(len(rAtk)):
            total_recall[i] = (total_recall[i] + rAtk[i])

        print(pAtK)
        print(rAtk, "\n")

    print("Process: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(query_process_time),sum(query_process_time)/len(query_process_time), len(query_process_time)))
    print("Query: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(query_time),sum(query_time)/len(query_time), len(query_time)))  

    print(total_precision)
    print(total_recall)

    average_precision = [x / 64  for x in total_precision]
    average_recall = [x / 64 for x in total_recall]

    plt.plot(scalex, average_precision)
    plt.plot(scalex, average_recall)

    plt.xlim(2, 10)
    plt.ylim(0, 1)
    plt.show()
