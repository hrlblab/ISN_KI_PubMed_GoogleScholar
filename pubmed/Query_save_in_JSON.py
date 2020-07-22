from pymed import PubMed
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import json


min_year = 2010
max_year = 2020
total = []
slope = []
number = 0
pre_json = []

json_file = './11figure.json'

label = ['Renal Pathology', 'Kidney Transplantation', 'Chronic kidney disease', 'Acute Kidney Injury',
         'Renal Insufficiency', 'renal hypotension', 'Drug Discovery', 'Immunology', 'Genetic', 'Geriatric', 'Cardiovascular disease']

temp_dict = {}

# Create a PubMed object that GraphQL can use to query
# Note that the parameters are not required but kindly requested by PubMed Central
# https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
pubmed = PubMed(tool="MyTool", email="quan.liu@vanderbilt.edu")

time = '(2010/1/1[Date - Publication]: 2020/12/31[Date - Publication])'
time_ml = '((2010/1/1[Date - Publication]: 2020/12/31[Date - Publication]) AND ("Artificial General Intelligence" OR "Artificial Intelligence" OR "Autoencoder" OR "auto encoder" OR "Reinforcement learning" OR "AI Governance" OR "Augmented Intelligence" OR "Decision Intelligence" OR "neural network" OR "Data Labeling" OR "Annotation Services" OR "Edge AI" OR "Smart Robotics" OR "Quantum Computing" OR "Digital Ethics" OR "AutoML" OR "Deep Neural" OR "Deep Learning" OR "Deep Network" OR "Convolutional Neural" OR "Graph Neural" OR "Generative Adversarial" OR "Adversarial Learning" OR "Natural Language Processing" OR "Recurrent Neural" OR "Computer Vision" OR "Cognitive Computing" OR "machine learning" OR "random forest" OR "support vector" OR "regression tree" OR "regression splines" OR "artificial neural" OR "Lasso" OR "decision tree" OR "linear regression" OR "bayesian" OR "regression model" OR "regression" OR "Supervised-learning" OR "clustering" OR "Dimensionality reduction" OR "Unsupervised-learning" OR "big-data" OR "data-mining" OR "semi-supervised" OR "self-learning" OR "sparse learning" OR "dictionary learning" OR "Feature learning" OR "Anomaly detection" OR "Robot learning" OR "algorithms" OR "Federated learning" OR "linear model" OR "pattern recognition" OR "information retrieval" OR "game theory" OR "information theory" OR "swarm intelligence" OR "Markov Decision" OR "Markov Random" OR "dynamic programming" OR "multilayer perceptrons" OR "component analysis" OR "Sparse coding" OR "subspace learning" OR "matrix factorization" OR "matrix decomposition" OR "NLP algorithm" OR "K means" OR "computer vision" OR "speech recognition" OR "predictive model" OR "machine learning"))'
kidney = '((2010/1/1[Date - Publication]: 2020/12/31[Date - Publication]) AND (("Glomeruli"[All Fields] OR "glomerular"[All Fields] OR "glomerulus"[All Fields] OR "glomerulosclerosis"[All Fields] OR "nephropathology"[All Fields] OR "renal pathology"[All Fields] OR "kidney pathology"[All Fields] OR "renal whole slide"[All Fields] OR "kidney whole slide"[All Fields] OR "renal wholeslide"[All Fields] OR "kidney wholeslide"[All Fields] OR "renal biopsy"[All Fields] OR "kidney biopsy"[All Fields] OR "Kidney/diagnostic imaging"[MAJR] OR "Kidney Glomerulus/pathology”[MAJR] OR "Kidney Diseases/pathology"[MAJR] OR "Kidney/pathology"[MAJR] OR ("Kidney"[MeSH] AND "Biopsy"[MeSH]) OR "Renal Dialysis"[MeSH] OR "Kidney Diseases"[MeSH] OR "Nephrology"[MeSH] OR "Nephrology" OR "Nephrologists"[MeSH] OR "Kidney"[MeSH] OR "Kidney Function Tests"[MeSH] OR "Kidney Function Tests"[MeSH] OR "Kidney Transplantation"[MeSH] OR "Hypertension, Renal"[MeSH] OR "Renal Insufficiency"[MeSH] OR "renal survival" OR "Acute kidney injury" OR "kidney transplantation" OR "kidney disease" OR "CKD" OR "AKI" OR "chronic kidney disease")))'


query_Renal_Pathology = '("Glomeruli"[All Fields] OR "glomerular"[All Fields] OR "glomerulus"[All Fields] OR "glomerulosclerosis"[All Fields] OR "nephropathology"[All Fields] OR "renal pathology"[All Fields] OR "kidney pathology"[All Fields] OR "renal whole slide"[All Fields] OR "kidney whole slide"[All Fields] OR "renal wholeslide"[All Fields] OR "kidney wholeslide"[All Fields] OR "renal biopsy"[All Fields] OR "kidney biopsy"[All Fields] OR "Kidney/diagnostic imaging"[MAJR] OR "Kidney Glomerulus/pathology"[MAJR] OR "Kidney Diseases/pathology"[MAJR] OR "Kidney/pathology"[MAJR] OR ("Kidney"[MeSH] AND "Biopsy"[MeSH]))'

query_Kidney_Transplantation = '(Kidney Transplantation)' + ' AND ' + kidney
query_CKD = '("chronic kidney disease" OR "CKD")' + 'AND' + kidney
query_Acute_Kidney_Injury = '("acute kidney injury" OR "AKI")' + 'AND' + kidney
query_Renal_Insufficiency = '("Renal Insufficiency")' + 'AND' + kidney
query_Renal_Hypotension = '("Hypertension, Renal" OR "renal hypertension")' + 'AND' + kidney
query_Drug_Discovery = '("Drug Discovery")' + ' AND ' + kidney
query_Immunology = '("Immunology")' + ' AND ' + kidney
query_Genetic = '("Genetic")' + ' AND ' + kidney
query_Geriatric = '("Geriatric")' + ' AND ' + kidney
query_Cardiovascular_disease = '("Cardiovascular disease")' + ' AND ' + kidney

query_ml_in_kidney = time_ml + ' AND ' + kidney
query_list = [ query_Renal_Pathology,
              query_Kidney_Transplantation, query_CKD, query_Acute_Kidney_Injury,
              query_Renal_Insufficiency, query_Renal_Hypotension,
              query_Drug_Discovery,
              query_Immunology,
              query_Genetic, query_Geriatric, query_Cardiovascular_disease]

    # Execute the query against the API
for query in query_list:
    print(str(query))
    result_1 = pubmed.query(query + 'AND' + time_ml, max_results=2000000)
    result_2 = pubmed.query(query, max_results=2000000)


    articleList = []
    articleInfo = []

    count_1 = np.zeros(11)
    count_2 = np.zeros(11)

    # Loop over the retrieved articles
    for article in result_1:
        articleDict = article.toDict()
        articleList.append(articleDict)

    for article in articleList:

        pubmedId = article['pubmed_id'].partition('\n')[0]
        # Append article info to dictionary
        articleInfo.append({u'pubmed_id': pubmedId,
                            u'title': article['title'],
                            u'abstract': article['abstract'],
                            u'publication_date': article['publication_date'],
                            u'authors': article['authors']})
        # print(type(article['publication_date']))
        # print(article['publication_date'])
        if isinstance(article['publication_date'], str):
            print("1")
        else:
            if article['publication_date'].year - min_year > -1:
                count_1[article['publication_date'].year - min_year] = count_1[article['publication_date'].year - min_year] + 1

    print(sum(count_1))
    print(count_1)

    articleList = []
    articleInfo = []

    # Loop over the retrieved articles
    for article in result_2:
        articleDict = article.toDict()
        articleList.append(articleDict)

    for article in articleList:
        # print(article)
        # Sometimes article['pubmed_id'] contains list separated with comma - take first pubmedId in that list - thats article pubmedId
        pubmedId = article['pubmed_id'].partition('\n')[0]
        # Append article info to dictionary
        articleInfo.append({u'pubmed_id': pubmedId,
                            u'title': article['title'],
                            u'abstract': article['abstract'],
                            u'publication_date': article['publication_date'],
                            u'authors': article['authors']})
        # print(type(article['publication_date']))
        # print(article['publication_date'])
        if isinstance(article['publication_date'], str):
            print("Publication date out of range")
        else:
            if article['publication_date'].year - min_year > -1:
                count_2[article['publication_date'].year - min_year] = count_2[article['publication_date'].year - min_year] + 1

    print(sum(count_2))
    print(count_2)

    name_list = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    x = list(range(len(count_1)))
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, count_1, width=width, label=label[number], fc='b')
    for a, b in zip(x, count_1):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, count_2, width=width, label='Machine learning in ' + label[number], tick_label=name_list, fc='y')

    for a, b in zip(x, count_2):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

    plt.ylabel('Number of Papers', fontsize=10)
    plt.title(label[number] + 'in Nephrology', fontsize=10)
    plt.xlabel('Year', fontsize=10)

    bottom, top = plt.ylim()  # return the current ylim
    plt.ylim((bottom, top+10))  # set the ylim to bottom, top

    plt.legend()
    plt.savefig(label[number] + '.png')
    plt.show()


    p = 0
    pre_json = []
    submit = './' + label[number] + '.json'

    for name in name_list:
        temp_dict = {}
        temp_dict[name] = [count_1[p], count_2[p]]
        pre_json.append(temp_dict)
        p = p + 1

    with open(submit, 'w') as f:
        json.dump(pre_json, f)

    number = number + 1