from utils import *
import time
import numpy as np
import xml.etree.ElementTree as ET
import json
source = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/source-document'
suspect = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document'
json_path = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/papers.json'

if __name__ == '__main__':
    engine = input("Digite 1 para whoosh e 2 para elasticsearch: ")

    ## ------------------------------------------------
    ##                 WHOOSH
    ## ------------------------------------------------
    if(engine == '1'):
        files = folderReader(source)

        index_process_time, index_time = indexDoc(files)

        print("Process index: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(index_process_time),sum(index_process_time)/len(index_process_time), len(index_process_time)))
        print("Process index: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(index_time),sum(index_time)/len(index_time), len(index_time)))

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


            # Extract subqueries
            subqueries = extractSubQueries(p_text, 2)

            # print(subqueries)

            query = whoosh.query.DisjunctionMax(subqueries)

            results = searchDoc(query)
            query_time.append(time.time() - start_preprocess)

            pAtK = []
            rAtk = []
            scalex = []
            hits = 0
            for i in range(len(results[:10])):  
                for doc in data[2]['src_file']:
                    if(results[i][0]['reference'] == doc):
                        hits += 1
                if i == 1 or i == 3 or i == 5 or i == 7 or i == 9:
                    pAtK.append(hits/(i+1)) 
                    rAtk.append(hits/len(data[2]['src_file']))
                    scalex.append(i+1)

            for i in range(len(pAtK)):
                total_precision[i] = (total_precision[i] + pAtK[i])
            
            for i in range(len(rAtk)):
                total_recall[i] = (total_recall[i] + rAtk[i])

        print("Process: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(query_process_time),sum(query_process_time)/len(query_process_time), len(query_process_time)))
        print("Query: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(query_time),sum(query_time)/len(query_time), len(query_time)))  

        average_precision = [x / 64  for x in total_precision]
        average_recall = [x / 64 for x in total_recall]

        plt.plot(scalex, average_precision)
        plt.plot(scalex, average_recall)

        plt.xlim(2, 10)
        plt.ylim(0, 1)
        plt.show()

    # ------------------------------------------------
    #                 ELASTICSEARCH
    # ------------------------------------------------
    elif(engine == '2'):
    
        files = folderReader(source)

        index_process_time, index_time = indexDocElastic(files)

        print("Process index: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(index_process_time),sum(index_process_time)/len(index_process_time), len(index_process_time)))
        print("Process index: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(index_time),sum(index_time)/len(index_time), len(index_time)))

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

            # Extract subqueries
            subqueries = extractSubQueriesElastic(p_text, 2)

            results = searchDocElastic(subqueries[:18000])

            query_time.append(time.time() - start_preprocess)

            pAtK = []
            rAtk = []
            scalex = []
            hits = 0
            for j in range(len(results[:10])):  
                for doc in data[i]['src_file']:
                    if(results[j]['_source']['reference'] == doc):
                        hits += 1
                if j == 1 or j == 3 or j == 5 or j == 7 or j == 9:
                    pAtK.append(hits/(j+1)) 
                    rAtk.append(hits/len(data[i]['src_file']))
                    scalex.append(j+1)

            for j in range(len(pAtK)):
                total_precision[j] = (total_precision[j] + pAtK[j])
            
            for j in range(len(rAtk)):
                total_recall[j] = (total_recall[j] + rAtk[j])

        print("Process: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(query_process_time),sum(query_process_time)/len(query_process_time), len(query_process_time)))
        print("Query: Tempo total: %2.5f, Tempo medio: %2.5f de %d docs armazenados "%(sum(query_time),sum(query_time)/len(query_time), len(query_time)))  

        average_precision = [x / 64  for x in total_precision]
        average_recall = [x / 64 for x in total_recall]

        plt.plot(scalex, average_precision, marker='o')
        plt.plot(scalex, average_recall, marker='o')

        plt.xlim(2, 10)
        plt.ylim(0, 1)
        plt.show()
    