import os

if not os.path.exists("engine/generation/data"):
    os.makedirs("engine/generation/data")

import ir_datasets
dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

for doc in dataset.docs_iter():
    doc_id = doc.doc_id
    content = doc.summary
    with open(f"engine/generation/data/{doc_id}.txt", "w") as out:
        out.write(content)