import pandas as pd
import glob    

# Reading all text files and writing to single processing file
path = r"ALT-Parallel-Corpus-20191206/"
f = open('outputFile.txt', 'w') 
for files in glob.glob(path +"*.txt"):
    lines = open(files,"r").readlines()
    for line in lines:
        f.write(line.strip() + '\t' + str(files).strip()+'\n')   
f.close()

my_cols = ["SID", "Sent", "lang"]
alt_dataset = pd.read_csv('outputFile.txt', sep='\\t', names=my_cols, engine='python')
alt_dataset['lang'] = alt_dataset['lang'].str.split('_').str[-1].str.split('.').str[0]

# Change file name here
alt_dataset.to_csv('data/en_yy.csv',sep='|', index=False)
