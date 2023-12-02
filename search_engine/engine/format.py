# Buat memproses file collection MED.ALL

# input = open("MED.ALL", "r")
# input_text = input.readlines()

# flag_awal = False
# for i in input_text:
#     if i.startswith('.I '):
#         index = i.split()[-1]
#         if flag_awal:
#             with open(f"data/{int(index)-1}.txt", 'w') as output_file:
#                 output_file.write(content)
#         flag_awal = True
#         content = ''
            
#     else:
#         if i.startswith('.W'):
#             continue
#         content += i.strip()

# input.close()

import os

if not os.path.exists('data'):
    os.makedirs('data')

with open("nfcorpus/test.docs") as file:
    for line in file:
        doc_id, content = line.split("\t")
        with open(f"data/{doc_id}.txt", 'w') as out:
            out.write(content)
    
