import numpy as np
import matplotlib.pyplot as plt

plt.axvline(x=0.33, c="r", ls="-", lw=1, label = '${t_1}$')

plt.axvline(x=0.66, c="b", ls="-", lw=1, label = '${t_3}$')

plt.axhline(y=0.5, xmin=0, xmax=0.33, c="g", ls="-", lw=1, label = '${t_2}$')

plt.axhline(y=0.5, xmin=0.66, xmax=1, c="y", ls="-", lw=1, label = '${t_4}$')

plt.text(0.07,0.25,"${R_1}$ and ${c_1}$",fontdict={'size':'16','color':'g'})

plt.text(0.07,0.75,"${R_2}$ and ${c_2}$",fontdict={'size':'16','color':'r'})

plt.text(0.4,0.5,"${R_3}$ and ${c_3}$",fontdict={'size':'16','color':'b'})

plt.text(0.73,0.25,"${R_4}$ and ${c_4}$",fontdict={'size':'16','color':'y'})

plt.text(0.73,0.75,"${R_5}$ and ${c_5}$",fontdict={'size':'16','color':'c'})

plt.xlabel('${X_1}$',fontdict={'weight': 'normal', 'size': 15})

plt.ylabel('${X_2}$',fontdict={'weight': 'normal', 'size': 15})

num1 = 1.05
num2 = 0
num3 = 3
num4 = 0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

plt.show()
