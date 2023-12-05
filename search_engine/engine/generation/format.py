import os

if not os.path.exists("engine/generation/data"):
    os.makedirs("engine/generation/data")

dataset = os.path.join(os.getcwd(), "engine/training/nfcorpus/test.docs")

with open(dataset) as file:
    for line in file:
        doc_id, content = line.split("\t")
        with open(f"engine/generation/data/{doc_id}.txt", "w") as out:
            out.write(content)
    