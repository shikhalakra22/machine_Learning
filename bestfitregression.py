from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,6,4,7,5,6], dtype=np.float64)

def best_fit(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) /  
           ((mean(xs)*mean(xs)) - mean(xs**2) ))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = best_fit(xs,ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color = 'g')
plt.plot(xs, regression_line)
plt.show()

print(m,b)

