# ISN_KI_PubMed_GoogleScholar

## Instruction of generating Figure 1
**Query Information from Pubmed**

Run 'Query_save_in_JSON.py'. Script is used to query related papers' information from Pubmed. Query results will be saved in json file.

**Draw scatter based on query result**

Run 'Draw_scatter_plot.py'. Scatter plot shows the relationship between growth rate and number of publications.

**Draw histogram based on query result**

Run 'Draw_histogram.py'. Histogram will be saved in eps format for further editing. 

## Instruction of generating Figure 2
**Step 1** 

Use "Publish or Perish" to query the publications'(https://harzing.com/resources/publish-or-perish) information. Save the results in json files. The json file name is the key word (label) that you want to display in the figure.

**Step 2** 

Run the script to get the growth rate and total numbers of publications for each label, then display them in the figure. The figure includes two parts. Figure (a) shows the labels whose total number is larger than a threshold. Figure (b) shows the rest of the labels and clusters them into four groups.

## Instruction of generating Figure 3
**Step 1** 

Use "Publish or Perish" to query the publications'(https://harzing.com/resources/publish-or-perish) information. Save the results in json files. The json file name is the key word (label) that you want to display in the figure.

**Step 2**

Run the script to calculates total numbers of publications in each year and display them in Figure(a). Then it gets the growth rate and total numbers of publications for each label, then displays them in the Figure (b).
