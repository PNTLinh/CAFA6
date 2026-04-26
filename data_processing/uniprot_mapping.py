"""
UniProt ID Mapping Script
--------------------------
Input : 9606.protein.links.v12.0.txt.gz  (STRING database)
Output: uniprot_ensembl_mapping.csv
Columns: UniProtKB_AC, Ensembl_Protein

Workflow:
  1. Extract unique ENSP IDs from the STRING file.
  2. Submit a job to the UniProt ID Mapping REST API
     (From: Ensembl_Protein  →  To: UniProtKB)
  3. Poll until the job finishes, then download + save as CSV.
"""

import gzip
import csv
import time
import json
import requests
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE  = Path(r"D:\CAFA6\9606.protein.links.v12.0.txt.gz")
OUTPUT_FILE = Path(r"D:\CAFA6\uniprot_ensembl_mapping.csv")

UNIPROT_API  = "https://rest.uniprot.org"
BATCH_SIZE   = 500      # UniProt recommends ≤ 500 IDs per batch
POLL_SECONDS = 5        # seconds between status checks
MAX_RETRIES  = 60       # give up after MAX_RETRIES * POLL_SECONDS seconds

# ── Step 1: Extract unique ENSP IDs ────────────────────────────────────────────
print("Step 1 – Reading STRING file and extracting unique ENSP IDs …")

ensp_ids: set[str] = set()
with gzip.open(INPUT_FILE, "rt") as fh:
    next(fh)                               # skip header
    for line in fh:
        p1, p2, _ = line.rstrip().split()
        # Strip taxonomy prefix: "9606.ENSP…" → "ENSP…"
        ensp_ids.add(p1.split(".", 1)[1])
        ensp_ids.add(p2.split(".", 1)[1])

ensp_list = sorted(ensp_ids)
print(f"  Found {len(ensp_list):,} unique Ensembl Protein IDs.")


# ── Helper functions ────────────────────────────────────────────────────────────
def submit_batch(ids: list[str]) -> str:
    """Submit a mapping job and return the job ID."""
    response = requests.post(
        f"{UNIPROT_API}/idmapping/run",
        data={
            "ids":  ",".join(ids),
            "from": "Ensembl_Protein",
            "to":   "UniProtKB",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["jobId"]


def wait_for_job(job_id: str) -> None:
    """Poll until the job is FINISHED (or raise on failure)."""
    url = f"{UNIPROT_API}/idmapping/status/{job_id}"
    for attempt in range(MAX_RETRIES):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("jobStatus", "")
        if status == "FINISHED":
            return
        if status == "FAILED":
            raise RuntimeError(f"Job {job_id} FAILED: {data}")
        # Still running – wait
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Job {job_id} did not finish within the timeout period.")


def fetch_results(job_id: str) -> list[dict]:
    """Download all result pages for a finished job."""
    results = []
    url = f"{UNIPROT_API}/idmapping/uniprotkb/results/{job_id}?format=tsv&fields=accession&size=500"
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        # TSV body
        lines = r.text.strip().splitlines()
        if lines:
            header = lines[0].split("\t")    # e.g. ["From", "Entry"]
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    results.append({
                        "Ensembl_Protein": parts[0],   # original ENSP ID
                        "UniProtKB_AC":    parts[1],   # UniProt accession
                    })

        # Pagination: follow "Link: <…>; rel="next"" header
        link_header = r.headers.get("Link", "")
        url = None
        if 'rel="next"' in link_header:
            # Extract the URL between < >
            start = link_header.index("<") + 1
            end   = link_header.index(">")
            url   = link_header[start:end]

    return results


# ── Step 2: Submit batches & collect results ────────────────────────────────────
print("\nStep 2 – Submitting ID mapping jobs to UniProt …")

all_results: list[dict] = []
total_batches = (len(ensp_list) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(total_batches):
    batch = ensp_list[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
    print(f"  Batch {batch_num + 1}/{total_batches}  ({len(batch)} IDs) … ", end="", flush=True)

    job_id = submit_batch(batch)
    wait_for_job(job_id)
    batch_results = fetch_results(job_id)
    all_results.extend(batch_results)
    print(f"→ {len(batch_results)} mappings found.")

print(f"\n  Total mappings collected: {len(all_results):,}")


# ── Step 3: Write CSV ───────────────────────────────────────────────────────────
print(f"\nStep 3 – Writing CSV to {OUTPUT_FILE} …")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["UniProtKB_AC", "Ensembl_Protein"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"  Done! {len(all_results):,} rows written to:\n  {OUTPUT_FILE}")
