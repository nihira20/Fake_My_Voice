import numpy as np
import pandas as pd
import os
import nltk
import re

datapath ="your_dataset_path"

nltk.download('averaged_perceptron_tagger_eng')
# !pip install g2p-en

metadata= pd.read_csv(datapath+"metadata.csv",sep="|", header = None)
metadata.columns = ["id", "transcript", "normalized"]

# print(metadata.shape)
# print(metadata.head(5))

row = metadata.iloc[0]

wavpath= os.path.join(datapath, "wavs", row["id"]+".wav")
text = row["normalized"]

print("audio file: ", wavpath)
print("text: ", text)

metadata["normalized"] = metadata["normalized"].str.lower()
pd.set_option("display.max_colwidth", None)


def cleantext(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9\s\']", "", text)
    return text

metadata["cleaned"] = metadata["normalized"].apply(cleantext)
metadata[["normalized","cleaned"]].head(2)
