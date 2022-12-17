#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('data2.csv')
df.head(7)


# In[3]:


df.drop(['id', 'track_name', 'genres_list'], axis=1, inplace=True)


# In[4]:


df["artist_name"] = (df["artist_name"].str.strip()).str.lower()
df["genres"] = (df["genres"].str.strip()).str.lower()


# In[5]:


columns = ['artist_name', 'subjectivity', 'polarity']
le = LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])


# In[6]:


df["loudness"] = df["loudness"] + 60


# In[7]:


columns = ['artist_name', 'key', 'artist_pop', 'track_pop', 'subjectivity', 'polarity']
for col in columns:
    df[col] = df[col] / df[col].max()


# In[8]:


columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for col in columns:
    df[col] = df[col] / df[col].max()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


# genre_list = []
# for genres in df['genres']:
#     genres = genres.split(' ')
#     for genre in genres:
#         genre_list.append(genre)


# In[12]:


# cnt = 0
# for i, genre in enumerate(genre_list):
#     if 'trap' in genre:
#         cnt += 1
# cnt


# In[13]:


main_genres = ['contemporary', 'country', 'dance', 'folk', 'hip_hop',
               'house', 'indie', 'metal', 'modern', 'pop',
               'rap', 'rock', 'r&b', 'soul', 'trap'
               ]


# In[14]:


genre_to_class = { 'other' : 0,
                   'contemporary' : 1,
                   'country' : 2,
                   'dance' : 3,
                   'folk' : 4,
                   'hip_hop' : 5,
                   'house' : 6,
                   'indie' : 7,
                   'metal' : 8,
                   'modern' : 9,
                   'pop' : 10,
                   'rap' : 11,
                   'rock' : 12,
                   'r&b' : 13,
                   'soul' : 14,
                   'trap' : 15
}


# In[15]:


class_to_genre = { 0 : 'other',
                   1 : 'contemporary',
                   2 : 'country',
                   3 : 'dance',
                   4 : 'folk',
                   5 : 'hip_hop',
                   6 : 'house',
                   7 : 'indie',
                   8 : 'metal',
                   9 : 'modern',
                   10 : 'pop',
                   11 : 'rap',
                   12 : 'rock',
                   13 : 'r&b',
                   14 : 'soul',
                   15 : 'trap',
}


# In[16]:


genre_classes = np.zeros((df.shape[0], len(genre_to_class)))
for i, genres in enumerate(df['genres']):
    genres = genres.split(' ')
    for genre in genres:
        other = True
        for main_genre in main_genres:
            if main_genre in genre:
                genre_classes[i, genre_to_class[main_genre]] = 1
                other = False
        if other:
            genre_classes[i, genre_to_class['other']] = 1


# In[17]:


df.drop(['genres'], axis=1, inplace=True)
number_features = 16


# In[18]:


df_train,df_test,y_train,y_test = train_test_split(df,genre_classes, test_size=0.15, random_state=12)


# In[ ]:





# In[19]:


x_train = df_train.to_numpy()
x_test = df_test.to_numpy()


# In[20]:


x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, test_size=0.1765, random_state=12)


# In[21]:


class spotifyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :]
        y = self.y[idx, :]
        return x,y


# In[22]:


dataset_train = spotifyDataset(x_train, y_train)
dataset_val = spotifyDataset(x_val, y_val)
dataset_test = spotifyDataset(x_test, y_test)
type(x_test)


# In[23]:


train_loader = DataLoader(dataset_train, batch_size = 1024, shuffle = True)
val_loader = DataLoader(dataset_val, batch_size = 1, shuffle = False)
test_loader = DataLoader(dataset_test, batch_size = 1, shuffle = False)


# In[24]:


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm1d(number_features)

        self.fc1 = nn.Linear(number_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(p=0.1)

        # self.fc3 = nn.Linear(64, 32)
        # self.bn3 = nn.BatchNorm1d(32)
        # self.drop3 = nn.Dropout(p=0.1)

        self.leakyrelU = nn.LeakyReLU()
        self.out = nn.Linear(64, len(genre_to_class))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn0(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leakyrelU(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leakyrelU(x)
        x = self.drop2(x)

        # x = self.fc3(x)
        # x = self.bn3(x)
        # x = self.leakyrelU(x)
        # x = self.drop3(x)

        x = self.out(x)
        x = self.sigmoid(x)
        return x


# In[25]:


model = MLP()
model = model.float()
import pickle


# In[26]:


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.8)


# In[27]:


model_name = './model.pt'


# In[ ]:





# In[28]:


model.load_state_dict(torch.load(model_name))
model.eval()
preds = None
targets = None
val_loss_best = 0
for i, (data, target) in enumerate(test_loader):
    output = model(data.float())
    pred = output.detach().numpy()
    loss = criterion(output, target.float())
    val_loss_best += loss.item()*data.size(0)

    if i==0:
        preds = pred
        targets = target.detach().cpu().numpy()
    else:
        preds = np.concatenate((preds, pred), axis=0)
        targets = np.concatenate((targets, target.detach().cpu().numpy()), axis=0)

#val_loss_best = val_loss_best / len(test_loader.dataset)
#val_loss_best


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


prediction_limit = 2


# In[30]:


CM = np.zeros((len(genre_to_class)+1, len(genre_to_class)+1))
for k in range(preds.shape[0]):
    indices = np.argsort(preds[k])
    for i in range(len(genre_to_class)-1, -1, -1):
        l = indices[i]
        if i+prediction_limit >= len(genre_to_class):
            if preds[k,l] >= 0.5:
                if targets[k,l] == 1:
                    CM[l,l] += 1
                else:
                    CM[len(genre_to_class),l] += 1
            else:
                if targets[k,l] == 1:
                    CM[l,len(genre_to_class)] += 1


# In[31]:


np.savetxt('confusion_matrix_limited.csv', CM, delimiter=',')


# In[32]:


100 * np.trace(CM) / np.sum(CM)


# In[33]:


hit_count = 0
for k in range(preds.shape[0]):
    indices = np.argsort(preds[k])
    for i in range(len(genre_to_class)-1, -1, -1):
        l = indices[i]
        if i+prediction_limit >= len(genre_to_class):
            if preds[k,l] >= 0.5:
                if targets[k,l] == 1:
                    hit_count += 1
                    break
100*hit_count/preds.shape[0]


# In[34]:


CMs = []
for l in range(len(genre_to_class)):
    CMs.append(np.zeros((2,2)))


# In[35]:


for k in range(preds.shape[0]):
    indices = np.argsort(preds[k])
    for i in range(len(genre_to_class)-1, -1, -1):
        l = indices[i]
        if i+prediction_limit >= len(genre_to_class):
            if preds[k,l] >= 0.5:
                if targets[k,l] == 1:
                    CMs[l][1,1] += 1
                else:
                    CMs[l][0,1] += 1
            else:
                if targets[k,l] == 1:
                    CMs[l][1,0] += 1
                else:
                    CMs[l][0,0] += 1


# In[36]:


accuracies = []
counts = []
for i in range(len(genre_to_class)):
    # print(CMs[i])
    counts.append(np.sum(CMs[i]))
    accuracies.append(100 * np.trace(CMs[i]) / np.sum(CMs[i]))
np.dot(counts, accuracies) / np.sum(counts)


# In[ ]:





# In[37]:


prediction_limit = 2


# In[38]:


CM = np.zeros((len(genre_to_class)+1, len(genre_to_class)+1))
for k in range(preds.shape[0]):
    indices = np.argsort(preds[k])
    hit = False
    for i in range(len(genre_to_class)-1, -1, -1):
        l = indices[i]
        if i+prediction_limit >= len(genre_to_class):
            if preds[k,l] >= 0.5:
                if targets[k,l] == 1:
                    CM[l,l] += 1
                    hit = True
                else:
                    CM[len(genre_to_class),l] += 1
    if not hit:
        for i in range(len(genre_to_class)-1, -1, -1):
            l = indices[i]
            if i+prediction_limit >= len(genre_to_class):
                if preds[k,l] < 0.5:
                    if targets[k,l] == 1:
                        CM[l,len(genre_to_class)] += 1


# In[39]:


np.savetxt('confusion_matrix_limited_stricted.csv', CM, delimiter=',')


# In[40]:


100 * np.trace(CM) / np.sum(CM)


# In[41]:


hit_count = 0
for k in range(preds.shape[0]):
    indices = np.argsort(preds[k])
    for i in range(len(genre_to_class)-1, -1, -1):
        l = indices[i]
        if i+prediction_limit >= len(genre_to_class):
            if preds[k,l] >= 0.5:
                if targets[k,l] == 1:
                    hit_count += 1
                    break
100*hit_count/preds.shape[0]


# In[42]:


CMs = []
for l in range(len(genre_to_class)):
    CMs.append(np.zeros((2,2)))


# In[43]:


for k in range(preds.shape[0]):
    indices = np.argsort(preds[k])

    hit = False
    for i in range(len(genre_to_class)-1, -1, -1):
        l = indices[i]
        if i+prediction_limit >= len(genre_to_class):
            if preds[k,l] >= 0.5:
                if targets[k,l] == 1:
                    CMs[l][1,1] += 1
                    hit = True
                else:
                    CMs[l][0,1] += 1
    if not hit:
        for i in range(len(genre_to_class)-1, -1, -1):
            l = indices[i]
            if i+prediction_limit >= len(genre_to_class):
                if preds[k,l] < 0.5:
                    if targets[k,l] == 1:
                        CMs[l][1,0] += 1
                    else:
                        CMs[l][0,0] += 1


# In[44]:


accuracies = []
counts = []
for i in range(len(genre_to_class)):
    # print(CMs[i])
    counts.append(np.sum(CMs[i]))
    accuracies.append(100 * np.trace(CMs[i]) / np.sum(CMs[i]))
#np.dot(counts, accuracies) / np.sum(counts)


# In[ ]:





# In[45]:


cnt = 1


# In[46]:


i = np.random.permutation(preds.shape[0])
xs = preds[i[0:cnt], :]


# In[47]:


for x in zip(xs):


    print('---PREDS---')
    print(np.argsort(x))
    indices = np.argsort(x)
    for i in range(2):
        print(class_to_genre[indices[0][0]])
        print(class_to_genre[indices[0][1]])

    print()


# In[48]:


type(df)


# In[49]:


model = pickle.load(open('model.pkl', 'rb'))


# In[50]:

import streamlit as st
#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import warnings

warnings.filterwarnings("ignore")
import pickle

# In[2]:


# In[4]:

artist_name = st.number_input('artist_name')
danceability = st.number_input('danceability')
energy = st.number_input('energy')
key = st.number_input('key')
loudness = st.number_input('loudness')
mode = st.number_input('mode')
speechiness = st.number_input('speechiness')
acousticness = st.number_input('acousticness')
instrumentalness = st.number_input('instrumentalness')
liveness = st.number_input('liveness')
valence = st.number_input('valence')
tempo = st.number_input('tempo')
artist_pop = st.number_input('artist_pop')
subjectivity = st.number_input('subjectivity')
polarity = st.number_input('polarity')

data = {'artist_name': artist_name,
        'danceability': danceability,
        'energy': energy,
        'key': key,
        'loudness': loudness,
        'mode': mode,
        'speechiness': speechiness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'valence': valence,
        'tempo': tempo,
        'artist_pop': artist_pop,
        'subjectivity': subjectivity,
        'polarity': polarity,
        'a':0
        }


# In[51]:


fea = pd.DataFrame(data, index=[0])
type(fea)
fea = fea.to_numpy()
type(fea)

st.subheader('Value')
a = st.button('value')
if a:

# In[54]:


    dataset_test = spotifyDataset(fea, fea)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)


    # In[ ]:





    # In[55]:


    preds = None
    targets = None
    val_loss_best = 0
    for i, (data, target) in enumerate(test_loader):
        output = model(data.float())
        pred = output.detach().numpy()


        if i==0:
            preds = pred
        else:
            preds = np.concatenate((preds, pred), axis=0)

    preds


    # In[56]:


cnt = 1


    # In[57]:


i = np.random.permutation(preds.shape[0])
xs = preds[i[0:cnt], :]


    # In[58]:


for x in zip(xs):


    print('---PREDS---')
    print(np.argsort(x))
    indices = np.argsort(x)
    for i in range(2):
        print(indices[0][1])
        st.write(class_to_genre[indices[0][i]])
        print()
        print()

    print()


    # In[61]:



    # In[ ]:




