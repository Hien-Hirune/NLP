"""
    Argument syntax:
        python demoNMF.py --input_file input.txt --result ouputNMF.txt
"""

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import sys

if len(sys.argv) != 5 or (sys.argv[1] != '--input_file' and sys.argv[3] != "result"):
    raise Exception("Argument syntax error!")

input_file = sys.argv[2]
output_file = sys.argv[4]

#Get the file
with open(input_file, 'r', encoding='UTF-8') as f:
    data = [s.strip() for s in f.readlines() if s != '\n']


#Get stop words
with open('stop_words.txt', 'r', encoding='UTF-8') as f:
    stop_word = [s.rstrip() for s in f.readlines()]

stop_word = stop_word + list('0123456789%@$.,=+-?!;/()*"&^:#|\n\t\'')

#STEP 1: Tokenized data
token_data = []
for sentence in data:
    text_token = ViTokenizer.tokenize(sentence)
    token_data.append(text_token)


#STEP 2: Defining the vectorizer
vectorizer = TfidfVectorizer(max_features= 1000, smooth_idf=False, stop_words = frozenset(stop_word))
matrix = vectorizer.fit_transform(token_data)
terms = vectorizer.get_feature_names()

#STEP 3: Using NMF
NMF_model = NMF(n_components=len(token_data))
NMF_model.fit(matrix)

#STEP 4: Iterating through each topic
f = open(output_file, 'w', encoding='UTF-8')
for idx, topic in enumerate(NMF_model.components_):
    f.write("Topic %d:\n" % (idx))        
    for i in topic.argsort()[:-7:-1]:
        f.write(terms[i] + " ")
    f.write('\n')
f.close()
        
