"""
    Argument syntax:
        python demoLSA.py --input_file input.txt --result ouputLSA.txt
"""

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sys

if len(sys.argv) != 5 or (sys.argv[1] != '--input_file' and sys.argv[3] != "result"):
    raise Exception("Argument syntax error!")

input_file = sys.argv[2]
output_file = sys.argv[4]

#Get the file
with open(input_file, 'r', encoding='UTF-8') as f:
    data = [s.strip() for s in f.readlines() if s != '\n']
f.close()

#Get stop words
with open('stop_words.txt', 'r', encoding='UTF-8') as f:
    stop_word = [s.rstrip() for s in f.readlines()]
f.close()
stop_word = stop_word + list('0123456789%@$.,=+-!?;/()*"&^:#|\n\t\'')

#STEP 1: Tokenized data
token_data = []
for sentence in data:
    text_token = ViTokenizer.tokenize(sentence)
    token_data.append(text_token)

#STEP 2: Defining the vectorizer
vectorizer = TfidfVectorizer(max_features= 1000, smooth_idf=False, stop_words = frozenset(stop_word))
matrix = vectorizer.fit_transform(token_data)

# Getting the terms 
terms = vectorizer.get_feature_names()

#STEP 3: SVD represent documents and terms in vectors
SVD_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=100, random_state=42)
SVD_model.fit(matrix)


#STEP 4: Iterating through each topic
f = open(output_file, 'w', encoding='UTF-8')
for i, comp in enumerate(SVD_model.components_):
    terms_comp = zip(terms, comp)
    # sorting the 7 most important terms
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    f.write("Topic "+str(i)+":\n")
    # printing the terms of a topic
    for t in sorted_terms:
        f.write(t[0] + " ")
    f.write('\n')
f.close()
