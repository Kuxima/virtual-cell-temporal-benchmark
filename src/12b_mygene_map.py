# -*- coding: utf-8 -*-
"""Map human gene symbols to Ensembl IDs with a reproducible local cache."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

import requests
import scanpy as sc

from common import DATA_PROC, RESULTS


API_URL = "https://mygene.info/v3/query"


def query(session, sym):
    for attempt in range(3):
        try:
            response = session.get(
                API_URL,
                params={"q": sym, "fields": "symbol,ensembl.gene", "species": "human", "size": 5},
                timeout=10,
            )
            if response.status_code == 200:
                for hit in response.json().get("hits", []):
                    if hit.get("symbol") != sym:
                        continue
                    ensembl = hit.get("ensembl")
                    if isinstance(ensembl, dict):
                        ensembl = [ensembl]
                    if isinstance(ensembl, list):
                        candidates = sorted(
                            {str(item.get("gene", "")) for item in ensembl if str(item.get("gene", "")).startswith("ENSG")}
                        )
                        if candidates:
                            return sym, candidates[0]
                return sym, None
        except requests.RequestException:
            if attempt < 2:
                time.sleep(1)
    return sym, None


def main():
    cached = RESULTS / "sym2ensg_mygene.json"
    if cached.exists() and os.environ.get("FORCE_MYGENE_REFRESH", "0") != "1":
        print(f"使用已有 MyGene 映射缓存: {cached}")
        return

    adata = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad", backed="r")
    try:
        symbols = sorted(set(adata.var_names.astype(str)))
    finally:
        adata.file.close()
    print(f"待映射符号 {len(symbols)}", flush=True)

    session = requests.Session()
    mapping = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(query, session, symbol) for symbol in symbols]
        for index, future in enumerate(as_completed(futures), start=1):
            symbol, ensembl = future.result()
            if ensembl:
                mapping[symbol] = ensembl
            if index % 500 == 0:
                print(f"  {index}/{len(symbols)} 命中 {len(mapping)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"最终映射 {len(mapping)}/{len(symbols)} ({time.time() - t0:.0f}s)")
    cached.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved {cached}")


if __name__ == "__main__":
    main()
