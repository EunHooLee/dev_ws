from locale import DAY_1
from socketserver import DatagramRequestHandler
import urllib.request
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning
from tokenizers import Regex
from konlpy.tag import Mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt

# data download
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

def main():
    data = pd.read_table('ratings.txt')
    # print(data[:10])
    # print('Total number of data : {}'.format(np.size(data)))

    sample_data = data[:100]
    sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","",)    
    # print(sample_data[:10])

    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

    tokenizer = Mecab()
    tokenized = [];
    for sentence in sample_data['document']:
        temp = tokenizer.morphs(sentence)
        temp = [word for word in temp if not word in stopwords]

        tokenized.append(temp)

    # print(tokenized[:3])

    vocab = FreqDist(np.hstack(tokenized))
    # print('The size of set of vocabuary: {}'.format(len(vocab)))
    # print(vocab['재밌'])

    vocab_size = 500
    vocab = vocab.most_common(vocab_size)

    word_to_index = {word[0] : index+2 for index, word in enumerate(vocab)}
    word_to_index['pad'] = 1
    word_to_index['unk'] = 0

    encoded = []
    for line in tokenized:
        temp = []
        for w in line:
            try:
                temp.append(word_to_index[w])
            except KeyError:
                temp.append(word_to_index['unk'])
        
        encoded.append(temp)

    # print(encoded[:2])
    max_len = max(len(l) for l in encoded)
    # print(max_len) # max_len = 63

    for line in encoded:
        if len(line) < max_len:
            line += [word_to_index['pad']] * (max_len - len(line))

    # print(encoded[:3])
    print(vocab[:1])
    print(encoded)

    # vocab         : (단어, 전체 데이터에서 그 단어의 빈도수) 의 tuple 형태 데이터가 저장됨
    #               : 전체 데이터에서 중복되는 데이터가 사라지고, 단어집합을 생성
    # word_to_index : dictionary 형태로 key(단어), value(인덱스)가 저장됨,
    # encoded       : word_to_index 를 (one) padding 을 통해 각 문장의 길이를 통일 시킴

if __name__ == '__main__':
    main()