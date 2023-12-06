import os

if not os.path.exists("engine/generation/title"):
    os.makedirs("engine/generation/title")

import ir_datasets
dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

for doc in dataset.docs_iter():
    doc_id = doc.doc_id
    title = doc.title
    with open(f"engine/generation/title/{doc_id}.txt", "w") as out:
        out.write(title)