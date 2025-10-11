import nltk
# pip install gtp_en

from g2p_en import G2p
from basic_cleaners import metadata
g2p = G2p()

def g2p_phonemes(text):
    phonemes = g2p(text)
    phonemes = [ph for ph in phonemes if ph.isalpha() or ph[:-1].isalpha()]
    return phonemes

metadata["phonemes"] = metadata["cleaned"].apply(g2p_phonemes)
print(metadata["phonemes"].head(1))