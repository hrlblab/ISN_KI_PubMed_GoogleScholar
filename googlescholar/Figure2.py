import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import numpy as np
import json
import codecs
from scipy import stats


# Machine Learning result
threshold = 200
threshold2 = 0


# Input the query results from Google Scholar. Each json file's name is the label.
result_dir =  '/Users/DengRuining/Desktop/Meeting with Dr. Huo/MLMerge'
files = glob.glob(os.path.join(result_dir,"*.json"))


# We set a threshold to seperate the results to two parts according to the total number of publications.
# Figure (a) is labels with big total number of publications.
# Figure (b) is labels with small totoal number of publications.
df_ml_big = pd.DataFrame(columns=['label','counts','slope'])
df_ml_small = pd.DataFrame(columns=['label','counts','slope'])
row1 = 0
row2 = 0

already_ml = []
count_ml = np.zeros((11,1))

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
        if not (data[j]['title'] in already_ml):
            already_ml.append(data[j]['title'])
            try:
                count_ml[data[j]['year'] - 2010] = count_ml[data[j]['year'] - 2010] + 1
            except:
                print('error')
        else:
            print("%s has already been counted in total number" % (data[j]['title']))



        # Calculate the publications number for each label in each year from 2010 to 2020
        if not data[j]['title']  in already_js:
            already_js.append(data[j]['title'])
            try:
                count[data[j]['year']-2010] = count[data[j]['year']-2010] + 1
            except:
                print('No year information')
        else:
            continue

    # Calculate the total publication number for each label from 2010 to 2020
    # When calculate the growth rate, we only count 10 year's numbers from 2010 to 2019. We normalize the numbers from 0 to 10
    counts = sum(count)
    max_point = max(count[1:10])

    points = np.zeros((10,2))
    for j in range(10):
        points[j][0] = j/10
        points[j][1] = int(count[j])/max_point

    slope, intercept, r_value, p_value, std_err = stats.linregress(points)


    # We don't show label 'Machine Learning" and "Artificial Intelligence" in our figure.
    # But we calculate their publication numbers.
    if counts >= threshold and label != 'Machine Learning' and label != 'Artificial Intelligence':
        df_ml_big.loc[row1] = [label,counts,slope]
        row1 = row1 + 1
    elif counts < 1000 and label != 'Machine Learning' and label != 'Artificial Intelligence':
        df_ml_small.loc[row2] = [label, counts, slope]
        row2 = row2 + 1


plt.rcParams['axes.unicode_minus'] = False


# Get the big number label texts, X values, and Y values from the frame.
label_name = df_ml_big['label']
paper_counts = df_ml_big['counts']
grow = df_ml_big['slope']


# You can set your size number according to the counts number. We use the same size for all the labels.
#size = df_ml_big['counts'].astype(int) * 1

# Choose your own color to display the points
Colors=('#82DED9','#FA7052','#7D7DFF','#DDDDFF','#BEDB5C','#7D7DFF','#5CDBB8','#0000C6','#000079','#CEFFCE','#28FF28','#007500','#8C8C00','#FFB5B5','#FF0000','#CE0000','#750000')


# Draw the figure (a)
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 1, 1)
ax.scatter(paper_counts, grow,s=100, c=Colors,zorder=30)
ax.set_title("(a) Machine Learning Techniques in Renal Pathology (Overall)")
ax.set_xlabel("Total Number of Publications")
ax.set_ylabel("Growth Rate of Publications")

# Add each label on the figure
for i in range(len(label_name)):
    plt.text(paper_counts[i] * 1.01, grow[i] * 1.01, label_name[i],
             fontsize=10, color="r", style="italic", weight="light",
             verticalalignment='center', horizontalalignment='right', rotation=0)


# Get the small number label texts, X values, and Y values from the frame.
label_name = df_ml_small['label']
paper_counts = df_ml_small['counts']
grow = df_ml_small['slope']
size = df_ml_small['counts'].astype(int) * 1

# Draw all the points with gray in figure (a)
ax.scatter(paper_counts, grow, s = 100, color='#CFCFCF',zorder=30)

# Set the figure size and the background
plt.axis([-100, 5000, -0.5, 1.2])
ax.margins(0)
ax.axvspan(-100, threshold, facecolor='white', alpha=0.2,zorder=20)
ax.axvspan(threshold, 5000, facecolor='#ACAEA8', alpha=0.2,zorder=20)

# Save the figure in both png and eps format.
fig.savefig("Figure_ML_big.png")
plt.savefig("Figure_ML_big.eps",format='eps',dpi=1000)
plt.show()
plt.draw()


# Draw the figure (b)
fig = plt.figure(figsize=(8, 6))  # 新建画布
ax = plt.subplot(1, 1, 1)  # 子图初始化


# We divide the small number labels into four groups
Fields_points = pd.DataFrame(columns=['label','counts','slope'])
Applications_points = pd.DataFrame(columns=['label','counts','slope'])
Model_points = pd.DataFrame(columns=['label','counts','slope'])
Learning_points = pd.DataFrame(columns=['label','counts','slope'])

rowrow1 = 0
rowrow2 = 0
rowrow3 = 0
rowrow4 = 0

Fields = ['Natural Language Processing','Cognitive Computing','Game Theory','Information Theory','Computer Vision','Decision Intelligence','Quantum Computing','Augmented Intelligence']
Applications = ['Dimensionality Reduction','Annotation Services','Speech Recognition','Information Retrieval','Feature Learning','Data Labeling']
Model = ['Decision Tree','Regression Splines','Artificial Neural Network','Markov Random Field','Matrix Factorization','K means','LASSO','Support Vector Machine','Regression Tree','Random Forest','Sparse Coding','Swarm Intelligence','Markov Decision','Dynamic Programming']
Learning = ['Deep Learning','Self-learning','Supervised-learning','Dictionary Learning','Semi-supervised learning','Unsupervised-learning','Sparse Learning']


for i in range(len(df_ml_small)):
    if df_ml_small['label'].loc[i] in Fields:
        Fields_points.loc[rowrow1] = df_ml_small.loc[i]
        rowrow1 = rowrow1 + 1
    elif df_ml_small['label'].loc[i] in Applications:
        Applications_points.loc[rowrow2] = df_ml_small.loc[i]
        rowrow2 = rowrow2 + 1
    elif df_ml_small['label'].loc[i] in Model:
        Model_points.loc[rowrow3] = df_ml_small.loc[i]
        rowrow3 = rowrow3 + 1
    elif df_ml_small['label'].loc[i] in Learning:
        Learning_points.loc[rowrow4] = df_ml_small.loc[i]
        rowrow4 = rowrow4 + 1

paper_counts1 = Fields_points['counts']
paper_counts2 = Applications_points['counts']
paper_counts3 = Model_points['counts']
paper_counts4 = Learning_points['counts']

grow1 = Fields_points['slope']
grow2 = Applications_points['slope']
grow3 = Model_points['slope']
grow4 = Learning_points['slope']

size1 = 100

type1 = ax.scatter(paper_counts1, grow1, s=100, color='#EFE076',cmap=plt.cm.jet,zorder=30)
type2 = ax.scatter(paper_counts2, grow2, s=100, color='#7D7DFF',cmap=plt.cm.jet,zorder=30)
type3 = ax.scatter(paper_counts3, grow3, s=100, color='#FA7052',cmap=plt.cm.jet,zorder=30)
type4 = ax.scatter(paper_counts4, grow4, s=100, color='#82DED9',cmap=plt.cm.jet,zorder=30)

plt.axis([-5, 205, -0.5, 1.2])
ax.margins(0)
ax.axvspan(-5, 98, 0, 0.49, facecolor='#ACAEA8', alpha=0.2,zorder=20)
ax.axvspan(-5, 98, 0.51, 1, facecolor='#ACAEA8', alpha=0.2,zorder=20)
ax.axvspan(102, 205,  0, 0.49, facecolor='#ACAEA8', alpha=0.2,zorder=20)
ax.axvspan(102, 205, 0.51,1,facecolor='#ACAEA8', alpha=0.2,zorder=20)


# Add legends for four groups
ax.legend((type1, type2, type3, type4), ("Research Fields", "Applications", "Learning Models","Learning Strategies"), loc = 4,)


ax.set_title("(b) Machine Learning Techniques in Renal Pathology (<200 publications)")
ax.set_xlabel("Total Number of Publications")
ax.set_ylabel("Growth Rate of Publications")

label_name = df_ml_small['label']
paper_counts = df_ml_small['counts']
grow = df_ml_small['slope']

for i in range(len(label_name)):
    plt.text(paper_counts[i] * 1.01, grow[i] * 1.01, label_name[i],
             fontsize=10, color="black", style="italic", weight="light",
             verticalalignment='center', horizontalalignment='right', rotation=0)

fig.savefig("Figure_ML_small.png")
plt.savefig("Figure_ML_small.eps",format='eps',dpi=1000)
plt.show()
plt.draw()