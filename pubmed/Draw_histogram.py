import json
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

json_file = 'Renal Pathology.json '
i = 0

count_1 = np.zeros(11)
count_2 = np.zeros(11)

name_list = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']  # Year list

with open(json_file) as f:
    data = json.load(f)

for year in name_list:

    count_1[int(year) - 2010] = data[i][year][0]
    count_2[int(year) - 2010] = data[i][year][1]
    i = i + 1

x = list(range(len(count_1)))
total_width, n = 0.8, 2
width = total_width / n
plt.bar(x, count_1, width=width, label='AI in Kidney Disease', fc='b')
for a, b in zip(x, count_1):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, count_2, width=width, label='Kidney Disease', tick_label=name_list, fc='y')

for a, b in zip(x, count_2):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

plt.ylabel('Number of Papers', fontsize=10)
plt.title('AI in Renal Pathology', fontsize=10)
plt.xlabel('Year', fontsize=10)

bottom, top = plt.ylim()  # return the current ylim
plt.ylim((bottom, top+10))  # set the ylim to bottom, top

plt.legend()
plt.savefig("10_AI_in_RP.eps",format='eps',dpi=1000)
plt.show()