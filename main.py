from utils import zipfGrapgh, fileReader, folderReader, preProcess, indexDoc, preProcessTest, indexTest
import xml.etree.ElementTree as ET
source = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/source-document'
suspect = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document'

if __name__ == '__main__':
    files = folderReader(source)

    file = preProcessTest('Testing is testing and testing')

    vector = ['testing', 'is', 'testing', 'and', 'testing']

    indexTest(file)
    indexTest(vector)

    # indexDoc(files, schema)
