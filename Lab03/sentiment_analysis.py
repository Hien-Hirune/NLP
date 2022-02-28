"""
    Argument syntax:
        python sentiment_analysis.py --input "Môn học rất bổ ích" --result sentiment.txt
"""

import pandas as pd
import numpy as np
import re
import sys
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer

#Get stop word
stop_word = list('0123456789%@$.,=+-!?;/()*"&^:#|\n\t\'')

#Get data
folder = ['train', 'dev', 'test']
path = './UIT-VSFC/'
data = []
target = []
for i in range(3):
    sentiment = pd.read_csv(path + folder[i] + '/sentiments.txt', header=None)
    sentences = pd.read_csv(path + folder[i] + '/sents.txt', delimiter = "\n", header=None)
    data = data + sentences[0].tolist()
    target = target + sentiment[0].tolist()


#Get sentence
if len(sys.argv) != 5 or (sys.argv[1] != '--input_file' and sys.argv[3] != "--result"):
    raise Exception("Argument syntax error!")
input = sys.argv[2]
output_file = sys.argv[4]

#STEP 1: Tokenized data
token_data = [] #là 1 list chứa các câu đã được tokenize
for sentence in data:
    text_token = ViTokenizer.tokenize(sentence)
    text_token = re.sub('\d+\w', '', text_token) #loại bỏ các số
    token_data.append(text_token)

#STEP 2: Count vectorizer
vectorizer_train = CountVectorizer(stop_words = frozenset(stop_word))
matrix_train = vectorizer_train.fit_transform(token_data)
matrix_train.shape
# Getting the terms 
terms_train = vectorizer_train.get_feature_names_out()

#STEP 3: preprocess input
text_token = ViTokenizer.tokenize(input)
text_token = re.sub('\d+\w', '', text_token) #loại bỏ các số
entry = [x for x in text_token.split() if x not in stop_word]

#STEP 4: define class NaiveBayes
class NaiveBayesClassifier:
    def __init__(self, X, y):
        '''
        X = data train (in sparse matrix), y = target train
        '''
        self.df = pd.DataFrame(X.toarray(), columns= terms_train) #dataframe training
        self.df['target'] = y
        self.n_features = len(self.df.loc[0,:]) - 1
        

    def classify(self, entry):

        prob = []
        for i in range(3):
            sum_ = self.df[self.df['target'] == i].sum(axis=0) # là một core series chứa tổng số lần xuất hiện của các từ
            n_n = sum_[:-1].sum() #tổng số từ (kể cả lặp) trong văn bản, trừ cột cuối củng là cột target
            p_class_i = len(self.df[self.df['target'] == i])/len(self.df) #xác suất P(y)

            lst = []
            for x in entry: #duyệt từng từ trong câu cần predict
                if x in sum_.index: #nếu từ đó thuộc data thì lấy tỉ lệ ra
                    lst.append((sum_[x]+1)/(n_n + self.n_features))
                else:
                    lst.append(1/(n_n + self.n_features))
            #print(lst)
            prob.append(np.prod(lst)*p_class_i) #np.prod là hàm nhân các phần tử trong mảng
            
        max_prob = np.max(prob)
        #print("Class:", prob.index(max_prob), ".Probability:", max_prob)
        return prob.index(max_prob)

#STEP 5: Define model
nbc = NaiveBayesClassifier(matrix_train, target)

#STEP 6: predict
predict = nbc.classify(entry)

#STEP 7: save output to file
print(predict)
with open(output_file, 'w', encoding='UTF-8') as f:
    f.write(str(predict))
