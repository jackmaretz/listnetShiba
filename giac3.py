import RankNet
import ListNet
Model = RankNet.RankNet()
model = ListNet.ListNet()



import pandas as pd
import numpy as np

train =  pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

df = pd.concat((train,test))
df =df[["cosine_y","query","numWordsQuery","numWordsCV","numCommonWords","score"]]

X1 = train[["cosine_y","query","numWordsQuery","numWordsCV","numCommonWords"]]
y1 = train["score"]


grouped = df.groupby(by='query')

for i,j in grouped:
    if len(j)>6:
        print(len(j))
        y = j['score'].to_numpy().flatten()
        X = j[["cosine_y","query","numWordsQuery","numWordsCV","numCommonWords"]].to_numpy()
        model.fit(X, y)
        y_pred = model.predict(X)
        print(y_pred)
    else:
        print(i,"ricerca di merda")


#X = X1.to_numpy()
#y = y1.to_numpy()
#y = y.flatten()
#Model.fit(X, y)
#model.fit(X,y)
#print(Model)
#y_pred = model.predict(X)
#print(y_pred)