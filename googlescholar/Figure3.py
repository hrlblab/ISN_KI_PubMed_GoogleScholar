import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import numpy as np
import json
import codecs
from scipy import stats


#Deep Learning result
threshold = 0

# Input the query results from Google Scholar. Each json file's name is the label.
result_dir =  '/Users/DengRuining/Desktop/Meeting with Dr. Huo/DLMerge'
files = glob.glob(os.path.join(result_dir,"*.json"))


df_dl = pd.DataFrame(columns=['label','counts','slope'])
row = 0

already_dl = []
count_dl = np.zeros((11,1))

former2015 = pd.DataFrame(columns=['label','title','year'])
location = 0

# Load each json file and go through all the publications that you query.
for i in range(len(files)):
    already_js = []
    count = np.zeros((11,1))
    root = files[i]

    # Get the label from each json file's name. Load the data.
    label = os.path.basename(files[i]).replace(".json","").replace("_"," ").replace(":","/")
    data = json.load(codecs.open(root, 'r', 'utf-8-sig'))

    # Calculate the total paper number for machine learning (all labels) .
    # Ignore the papers that have been calculated before.
    for j in range(len(data)):
        if not (data[j]['title'] in already_dl):
            already_dl.append(data[j]['title'])
            try:
                count_dl[data[j]['year'] - 2010] = count_dl[data[j]['year'] - 2010] + 1
            except:
                print('error')
        else:
            print("%s has already been counted in total number" % (data[j]['title']))


        # Calculate the publications number for each label in each year from 2010 to 2020
        if not data[j]['title'] in already_js:
            already_js.append(data[j]['title'])
            try:
                count[data[j]['year'] - 2010] = count[data[j]['year'] - 2010] + 1
            except:
                print('No year information')
        else:
            continue

    # Calculate the total publication number for each label from 2010 to 2020
    # When calculate the growth rate, we only count 10 year's numbers from 2010 to 2019. We normalize the numbers from 0 to 10
    counts = sum(count)
    max_point = max(count[1:10])

    points = np.zeros((10, 2))
    for j in range(10):
        points[j][0] = j / 10
        points[j][1] = int(count[j]) / max_point

    slope, intercept, r_value, p_value, std_err = stats.linregress(points)

    # We don't show label 'Deep Learning" and some technical terms in our figure.
    # But we calculate their publication numbers.
    if counts >= threshold and label != 'Deep Learning' and label != 'Deep Learning new':
        df_dl.loc[row] =[label,counts,slope]
        row = row + 1



plt.rcParams['axes.unicode_minus'] = False  #


# Get the big number label texts, X values, and Y values from the frame.
label_name = df_dl['label']
paper_counts = df_dl['counts']
grow = df_dl['slope']


fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 1, 1)

# Choose your own color to display the points.
Colors=('#82DED9','#FA7052','#000079','#CEFFCE','#FFB5B5','#FF0000','#CE0000','#750000')

# Draw all the points in figure (b)
ax.scatter(paper_counts, grow, s=100, c=Colors,zorder=30)  # 绘制散点图
ax.set_title("(b) Deep Learning Techniques in Renal Pathology")
ax.set_xlabel("Total Number of Publications")
ax.set_ylabel("Growth Rate of Publications")

# Add each label on the figure
for i in range(len(label_name)):
    plt.text(paper_counts[i] * 1.01, grow[i] * 1.01, label_name[i],
             fontsize=10, color="r", style="italic", weight="light",
             verticalalignment='center', horizontalalignment='right', rotation=0)

# Set the figure size and the background
plt.axis([-5, 200, 0.5, 1.1])
ax.margins(0)
ax.axvspan(-5, 98, 0, 0.49, facecolor='#ACAEA8', alpha=0.2,zorder=20)
ax.axvspan(102, 200, 0, 0.49, facecolor='#ACAEA8', alpha=0.2,zorder=20)
ax.axvspan(-5, 98, 0.51, 1, facecolor='#ACAEA8', alpha=0.2,zorder=20)
ax.axvspan(102, 200, 0.51, 1,facecolor='#ACAEA8', alpha=0.2,zorder=20)

# Save the figure in both png and eps format.
fig.savefig("Figure_DL.png")
plt.savefig("Figure_DL.eps",format='eps',dpi=1000)
plt.show()
plt.draw()

#Draw the total number for deep learning in each year
num_list1 = [int(count_dl[0]), int(count_dl[1]), int(count_dl[2]), int(count_dl[3]), int(count_dl[4]),
             int(count_dl[5]), int(count_dl[6]),int(count_dl[7]), int(count_dl[8]), int(count_dl[9]), int(count_dl[10])]

year_list = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']

x = list(range(len(num_list1)))

total_width, n = 1.2, 2
width = total_width / n

x = np.arange(len(year_list))

fig, ax= plt.subplots()

#Get the each number in different years as a text
rects1 = ax.bar(x , num_list1, width/2, color='#5C7FDB')

ax.set_ylabel('Total Number of Publications')
ax.set_title('(a) Deep Learning Publications in Renal Pathology')
ax.set_xticks(x)
ax.set_xticklabels(year_list)


def autolabel(rects1):
    # Attach the number and show its height.
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)

# Save the figure in both png and eps format.
fig.savefig("Figure4_google_scholar.png")
plt.savefig("Figure4_google_scholar.eps",format='eps',dpi=1000)
plt.show()
plt.draw()