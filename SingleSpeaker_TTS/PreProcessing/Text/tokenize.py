from basic_cleaners import metadata

specialtokens = ["<PAD>", "<UNK>", "<SOS>","<EOS>"]
allphonemes = set()
for text in metadata["phonemes"]:
    allphonemes.update(text)

phonemelist = specialtokens+ sorted(list(allphonemes))

phonemetoid = {p: i for i, p in enumerate(phonemelist)}
idtophoneme = {i: p for p, i in phonemetoid.items()}

print(phonemetoid)


for i, (p, idx) in enumerate(phonemetoid.items()):
    if i<10:
        print(f"{p} : {idx}")

sos_id = phonemetoid["<SOS>"]
eos_id = phonemetoid["<EOS>"]
unk_id = phonemetoid["<UNK>"]

def tokenizetext(seq):
    return [sos_id]+[phonemetoid.get(ph, unk_id) for ph in seq]+[eos_id]

metadata["phoneme_ids"] = metadata["phonemes"].apply(tokenizetext)


print("Phonemes:", metadata["phonemes"].iloc[0])
print("IDs :", metadata["phoneme_ids"].iloc[0])