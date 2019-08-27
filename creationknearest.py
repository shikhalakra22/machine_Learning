import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k': [ [1,2],[2,3],[3,1]], 'r':[ [6,5],[7,7],[8,6]]}
new = [5,7]




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

result = knear(dataset, new, k=3)
print(result)
       
for i in dataset:
    for j in dataset[i]:
        [[plt.scatter(j[0],j[1], s=100, color=i) for j in dataset[i]] for i in dataset]
plt.scatter(new[0],new[1])
plt.show()
