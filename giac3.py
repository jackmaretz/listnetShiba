import RankNet
import ListNet
Model = RankNet.RankNet()
model = ListNet.ListNet()



import pandas as pd
import numpy as np

train =  pd.read_csv("train.csv")

X1 = train[["cosine_y","query","numWordsQuery","numWordsCV","numCommonWords"]]
y1 = train["score"]

X = X1.to_numpy()
y = y1.to_numpy()
y = y.flatten()
Model.fit(X, y)
#model.fit(X,y)
#print(Model)
y_pred = model.predict(X)
print(y_pred)