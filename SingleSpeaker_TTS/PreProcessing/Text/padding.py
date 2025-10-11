from basic_cleaners import metadata
from tokenize import phonemetoid
import numpy as np

pad_id= phonemetoid["<PAD>"]#0
maxlength= 0
for token in metadata["phoneme_ids"]:
    if len(token) > maxlength:
        maxlength = len(token)

print(maxlength)

def padsequence(token, maxlength, pad_id):
    return token+[pad_id]*(maxlength-len(token))

metadata["phoneme_id_padded"] = metadata["phoneme_ids"].apply(lambda x: padsequence(x,maxlength, pad_id))

for i in range(2):
    print(metadata["phoneme_id_padded"][i])

phoneme_ids_array = np.array(metadata["phoneme_id_padded"].tolist())