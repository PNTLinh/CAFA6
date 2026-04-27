import gzip
import json
go_annotations = {}
with gzip.open("D:/CAFA6/goa_human.gaf.gz", "rt") as f:
    for line in f:
        if line.startswith("!"):
            continue  # 跳过注释行
        cols = line.strip().split("\t")
        protein_id = cols[1]  # UniProt ID
        go_id = cols[4]  # GO Term
        if protein_id not in go_annotations:
            go_annotations[protein_id] = []
        go_annotations[protein_id].append(go_id)

with open("D:/CAFA6/proceed_data/HUMAN_protein_info.json", "w") as f:
    json.dump(go_annotations, f, indent=4)
