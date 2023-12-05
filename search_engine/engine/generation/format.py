import os

if not os.path.exists("search_engine/engine/generation/data"):
    os.makedirs("search_engine/engine/generation/data")

dataset = os.path.join(os.getcwd(), "search_engine/engine/training/nfcorpus/test.docs")

with open(dataset) as file:
    for line in file:
        doc_id, content = line.split("\t")
        with open(f"search_engine/engine/generation/data/{doc_id}.txt", "w") as out:
            out.write(content)
    