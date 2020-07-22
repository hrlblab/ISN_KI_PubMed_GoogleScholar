from pymed import PubMed
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import json

def f_1(x, A, B):
    return A*x + B

min_year = 2010
max_year = 2020
total = []
slope = []
label = ['Renal Pathology', 'Kidney Transplantation', 'Chronic kidney disease', 'Acute Kidney Injury',
         'Renal Insufficiency', 'renal hypotension', 'Drug Discovery', 'Immunology', 'Genetic', 'Geriatric', 'Cardiovascular disease']

# Create a PubMed object that GraphQL can use to query
# Note that the parameters are not required but kindly requested by PubMed Central
# https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
pubmed = PubMed(tool="MyTool", email="quan.liu@vanderbilt.edu")

# query terms
time_ml = '((2010/1/1[Date - Publication]: 2020/12/31[Date - Publication]) AND ("Artificial General Intelligence" OR "Artificial Intelligence" OR "Autoencoder" OR "auto encoder" OR "Reinforcement learning" OR "AI Governance" OR "Augmented Intelligence" OR "Decision Intelligence" OR "neural network" OR "Data Labeling" OR "Annotation Services" OR "Edge AI" OR "Smart Robotics" OR "Quantum Computing" OR "Digital Ethics" OR "AutoML" OR "Deep Neural" OR "Deep Learning" OR "Deep Network" OR "Convolutional Neural" OR "Graph Neural" OR "Generative Adversarial" OR "Adversarial Learning" OR "Natural Language Processing" OR "Recurrent Neural" OR "Computer Vision" OR "Cognitive Computing" OR "machine learning" OR "random forest" OR "support vector" OR "regression tree" OR "regression splines" OR "artificial neural" OR "Lasso" OR "decision tree" OR "linear regression" OR "bayesian" OR "regression model" OR "regression" OR "Supervised-learning" OR "clustering" OR "Dimensionality reduction" OR "Unsupervised-learning" OR "big-data" OR "data-mining" OR "semi-supervised" OR "self-learning" OR "sparse learning" OR "dictionary learning" OR "Feature learning" OR "Anomaly detection" OR "Robot learning" OR "algorithms" OR "Federated learning" OR "linear model" OR "pattern recognition" OR "information retrieval" OR "game theory" OR "information theory" OR "swarm intelligence" OR "Markov Decision" OR "Markov Random" OR "dynamic programming" OR "multilayer perceptrons" OR "component analysis" OR "Sparse coding" OR "subspace learning" OR "matrix factorization" OR "matrix decomposition" OR "NLP algorithm" OR "K means" OR "computer vision" OR "speech recognition" OR "predictive model" OR "machine learning"))'
kidney = '(("Glomeruli"[All Fields] OR "glomerular"[All Fields] OR "glomerulus"[All Fields] OR "glomerulosclerosis"[All Fields] OR "nephropathology"[All Fields] OR "renal pathology"[All Fields] OR "kidney pathology"[All Fields] OR "renal whole slide"[All Fields] OR "kidney whole slide"[All Fields] OR "renal wholeslide"[All Fields] OR "kidney wholeslide"[All Fields] OR "renal biopsy"[All Fields] OR "kidney biopsy"[All Fields] OR "Kidney/diagnostic imaging"[MAJR] OR "Kidney Glomerulus/pathology”[MAJR] OR "Kidney Diseases/pathology"[MAJR] OR "Kidney/pathology"[MAJR] OR ("Kidney"[MeSH] AND "Biopsy"[MeSH]) OR "Renal Dialysis"[MeSH] OR "Kidney Diseases"[MeSH] OR "Nephrology"[MeSH] OR "Nephrology" OR "Nephrologists"[MeSH] OR "Kidney"[MeSH] OR "Kidney Function Tests"[MeSH] OR "Kidney Function Tests"[MeSH] OR "Kidney Transplantation"[MeSH] OR "Hypertension, Renal"[MeSH] OR "Renal Insufficiency"[MeSH] OR "renal survival" OR "Acute kidney injury" OR "kidney transplantation" OR "kidney disease" OR "CKD" OR "AKI" OR "chronic kidney disease"))'

query_Renal_Pathology = '("Glomeruli"[All Fields] OR "glomerular"[All Fields] OR "glomerulus"[All Fields] OR "glomerulosclerosis"[All Fields] OR "nephropathology"[All Fields] OR "renal pathology"[All Fields] OR "kidney pathology"[All Fields] OR "renal whole slide"[All Fields] OR "kidney whole slide"[All Fields] OR "renal wholeslide"[All Fields] OR "kidney wholeslide"[All Fields] OR "renal biopsy"[All Fields] OR "kidney biopsy"[All Fields] OR "Kidney/diagnostic imaging"[MAJR] OR "Kidney Glomerulus/pathology"[MAJR] OR "Kidney Diseases/pathology"[MAJR] OR "Kidney/pathology"[MAJR] OR ("Kidney"[MeSH] AND "Biopsy"[MeSH]))' + ' AND ' + time_ml
query_Kidney_Transplantation = '(Kidney Transplantation)' + ' AND ' + time_ml + ' AND ' + kidney
query_CKD = '("chronic kidney disease" OR "CKD")' + ' AND ' + time_ml + 'AND' + kidney
query_Acute_Kidney_Injury = '("acute kidney injury" OR "AKI")' + ' AND ' + time_ml + 'AND' + kidney
query_Renal_Insufficiency = '("Renal Insufficiency")' + ' AND ' + time_ml + 'AND' + kidney
query_Renal_Hypotension = '("Hypertension, Renal" OR "renal hypertension")' + ' AND ' + time_ml + 'AND' + kidney
query_Drug_Discovery = '("Drug Discovery")' + ' AND ' + kidney + ' AND ' + time_ml
query_Immunology = '("Immunology")' + ' AND ' + kidney + ' AND ' + time_ml
query_Genetic = '("Genetic")' + ' AND ' + kidney + ' AND ' + time_ml
query_Geriatric = '("Geriatric")' + ' AND ' + kidney + ' AND ' + time_ml
query_Cardiovascular_disease = '("Cardiovascular disease")' + ' AND ' + kidney + ' AND ' + time_ml

query_list = [query_Renal_Pathology, query_Kidney_Transplantation, query_CKD, query_Acute_Kidney_Injury,
              query_Renal_Insufficiency, query_Renal_Hypotension,
              query_Drug_Discovery,
              query_Immunology,
              query_Genetic, query_Geriatric, query_Cardiovascular_disease]

for query in query_list:
    results = pubmed.query(query, max_results=20000)
    articleList = []
    articleInfo = []

    count = np.zeros(11)

    # Loop over the retrieved articles
    for article in results:
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

        if isinstance(article['publication_date'], str):
            print("1")
        else:
            count[article['publication_date'].year - min_year] = count[article['publication_date'].year - min_year] + 1

    print(sum(count))
    print(count)

    A1, B1 = optimize.curve_fit(f_1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], count)[0]
    print(A1)

    total.append(sum(count))
    slope.append(A1)


plt.scatter(total, slope)
for i in range(len(total)):
    plt.annotate(label[i], xy = (total[i], slope[i]), xytext = (total[i]+0.3, slope[i]+0.3))   # add text and label on scatter plot
plt.show()


temp_dict = {}
p = 0
pre_json = []
submit = './data.json'    # save dictionary into a json file
for name in label:
    temp_dict[name] = [total[p], slope[p]]
    pre_json.append(temp_dict)
    p = p + 1
with open(submit, 'w') as f:
    json.dump(pre_json, f)

