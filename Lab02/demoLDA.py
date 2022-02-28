"""
    Argument syntax:
        python demoLDA.py --input_file input.txt --result ouputLDA.txt
"""

# Import gensim, nltk

from gensim import models, corpora
from pyvi import ViTokenizer
import sys
from nltk.tokenize import wordpunct_tokenize
from gensim.models.ldamodel import LdaModel

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

stop_word = stop_word + list('0123456789%@$.,=+-!?;/()*"&^:#|\n\t\'')

#STEP 1: Tokenized data
token_data = []
for sentence in data:
    text_token = ViTokenizer.tokenize(sentence)
    token_data.append(text_token)

#STEP 2: prepare a list containing lists of tokens of each text
all_tokens=[]
for text in token_data:
  tokens = []
  raw = wordpunct_tokenize(text)
  for token in raw:
    if token not in stop_word:
        tokens.append(token)
        all_tokens.append(tokens)

# STEP 3: Creating a gensim dictionary and the matrix
dictionary = corpora.Dictionary(all_tokens)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in all_tokens]

# STEP 4: Building the model and training it with the matrix 
model = LdaModel(doc_term_matrix, num_topics=len(token_data), id2word = dictionary, passes=40)
list_topic = model.print_topics(num_topics=len(token_data),num_words=6)
with open(output_file, 'w', encoding='UTF-8') as f:
  for i in range(len(list_topic)):
    f.write("Topic " + str(i) + ": " + list_topic[i][1] + "\n")
