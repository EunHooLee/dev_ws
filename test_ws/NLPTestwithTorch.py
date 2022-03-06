from re import I
from torchtext.legacy.data import TabularDataset
import urllib.request
import pandas as pd
from torchtext.legacy import data
from torchtext.legacy.data import Iterator

# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

# df = pd.read_csv('IMDb_Reviews.csv',encoding='latin1')
# print(df.head())

# train_df = df[:25000]
# test_df = df[25000:]

# train_df.to_csv("train_data.csv",index=False)
# train_df.to_csv("test_data.csv",index=False)

def main():
    TEXT = data.Field(sequential=True,
    use_vocab=True,
    tokenize=str.split,
    lower=True,
    batch_first=True,
    fix_length=20)
    
    LABEL = data.Field(sequential=False,
    use_vocab=False,
    batch_first=False,
    is_target=True)

    train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv',format='csv',
        fields = [('text',TEXT),('label',LABEL)], skip_header=True
    )

    # print(len(train_data), len(test_data))
    # print(vars(train_data[3]))

    TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

    batch_size = 5

    train_loader = Iterator(dataset=train_data, batch_size= batch_size)
    test_loader = Iterator(dataset=test_data,batch_size=batch_size)

    batch = next(iter(train_loader))
    # print(batch.text)
    # batch size 를 5 로 설정했고, 샘플의 길이를 20으로 설정했기 때문에 하나의 미니배치 크기는 5 x 20 matrix다.


if __name__ == '__main__':
    main()