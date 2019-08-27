import numpy as np
from math import sqrt
import pandas as pd
import warnings
from collections import Counter
import random



def knear(data, predict, k=3):
    if(len(data) >= k):
       warnings.warn('k is a set to a value less than total voting groups!')
    dist = []
    for group in data:
       for f in data[group]:
           distance = np.linalg.norm(np.array(f)-np.array(predict))
           dist.append([distance, group])

    votes = [i[1] for i in sorted(dist)[:k]]
    print(Counter(votes).most_common(1))
    vote_res = Counter(votes).most_common(1)[0][0]

    return vote_res

df = pd.read_csv("breastcancer.txt")
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[:-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = knear(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
c = float(correct)/total
print("Accuracy:", c)

