#!/usr/bin/env python3
"""
Push dei risultati verso saracandu/entropy-traces su HuggingFace Hub,
in BATCH (un commit ogni N file) per rispettare il rate limit sui commit
(128/ora), che con upload_file singolo si esaurisce quasi subito.

Famiglie di file:
  - "*_teacher_traces.json"              -> {dataset}/{fname}
  - "*_eval_sentence_*_nopatch.json"     -> {dataset}/results_sentence/{fname}
  - "*_eval_*.json" (senza "_sentence_") -> {dataset}/results/{fname}

Uso:
    python3 push_to_hf.py [data_dir] [--dry-run] [--batch-size N] [--start-at N]

Default: data_dir="data", batch_size=200
--start-at permette di riprendere da un punto preciso se il rate limit
colpisce di nuovo a meta' (indice nella lista ordinata dei file, 0-based).
"""
import sys
import time
from pathlib import Path


REPO_ID = "saracandu/entropy-traces"
REPO_TYPE = "dataset"


def classify(fname: str, dataset: str):
    if fname.endswith("_teacher_traces.json"):
        return f"{dataset}/{fname}"
    if "_eval_sentence_" in fname and fname.endswith(".json"):
        return f"{dataset}/results_sentence/{fname}"
    if "_eval_" in fname and fname.endswith(".json"):
        return f"{dataset}/results/{fname}"
    return None


def parse_args():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    batch_size = 500
    start_at = 0
    skip_big = 0
    skip_small_batches = 0
    if "--batch-size" in args:
        i = args.index("--batch-size")
        batch_size = int(args[i + 1])
    if "--start-at" in args:
        i = args.index("--start-at")
        start_at = int(args[i + 1])
    if "--skip-big" in args:
        i = args.index("--skip-big")
        skip_big = int(args[i + 1])
    if "--skip-small-batches" in args:
        i = args.index("--skip-small-batches")
        skip_small_batches = int(args[i + 1])
    consumed = {str(batch_size), str(start_at), str(skip_big), str(skip_small_batches)}
    positional = [a for a in args if not a.startswith("--") and a not in consumed]
    data_dir = Path(positional[0]) if positional else Path("data")
    return data_dir, dry_run, batch_size, start_at, skip_big, skip_small_batches


def main():
    data_dir, dry_run, batch_size, start_at, skip_big, skip_small_batches = parse_args()

    if not data_dir.exists():
        print(f"ERRORE: '{data_dir}' non trovata")
        sys.exit(1)

    to_upload = []
    for dataset_dir in sorted(data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for f in sorted(dataset_dir.iterdir()):
            if not f.is_file():
                continue
            path_in_repo = classify(f.name, dataset)
            if path_in_repo is None:
                continue
            to_upload.append((f, path_in_repo))

    print(f"{len(to_upload)} file totali trovati.")
    if start_at:
        to_upload = to_upload[start_at:]

    if dry_run:
        for local, remote in to_upload[:10]:
            print(f"  {local}  ->  {remote}")
        if len(to_upload) > 10:
            print(f"  ... e altri {len(to_upload) - 10}")
        print("\n[DRY RUN] Nessun upload eseguito.")
        return

    from huggingface_hub import HfApi, CommitOperationAdd
    import re
    api = HfApi()

    LARGE_THRESHOLD_BYTES = 5 * 1024 * 1024

    big_files = [(l, r) for l, r in to_upload if l.stat().st_size > LARGE_THRESHOLD_BYTES]
    small_files = [(l, r) for l, r in to_upload if l.stat().st_size <= LARGE_THRESHOLD_BYTES]

    if skip_big:
        print(f"Salto i primi {skip_big} file grandi (gia' caricati in run precedenti).")
        big_files = big_files[skip_big:]

    print(f"File grandi da caricare: {len(big_files)}")
    print(f"File piccoli da caricare: {len(small_files)}\n")

    # --- 1. file grandi, uno alla volta (ognuno = 1 commit, quindi vanno contati nel budget 128/ora) ---
    for i, (local, remote) in enumerate(big_files, 1):
        print(f"[grande {i}/{len(big_files)}] {remote} ({local.stat().st_size/1e6:.0f}MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print(f"  OK.")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                m = re.search(r"retry (?:this action )?in (\d+) minutes?", msg.lower())
                wait_hint = f"{m.group(1)} minuti" if m else "qualche minuto (vedi messaggio errore sopra)"
                print(f"\nRATE LIMIT SUI COMMIT. Aspetta {wait_hint}, poi riprendi con:")
                print(f"  python3 push_to_hf.py --skip-big {i-1}")
                sys.exit(1)
            else:
                print(f"  ERRORE non rate-limit: {msg[:300]}. Salto questo file.")
        time.sleep(2)

    # --- 2. file piccoli, in batch via create_commit (ognuno = 1 commit) ---
    n_batches = (len(small_files) + batch_size - 1) // batch_size
    for b in range(skip_small_batches, n_batches):
        chunk = small_files[b * batch_size: (b + 1) * batch_size]
        ops = [
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for local, remote in chunk
        ]
        print(f"Batch piccoli {b+1}/{n_batches} ({len(chunk)} file)...")
        try:
            api.create_commit(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=ops,
                commit_message=f"Upload small files batch {b+1}/{n_batches}",
            )
            print(f"  OK, commit creato.")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                m = re.search(r"retry (?:this action )?in (\d+) minutes?", msg.lower())
                wait_hint = f"{m.group(1)} minuti" if m else "qualche minuto (vedi messaggio errore sopra)"
                print(f"\nRATE LIMIT SUI COMMIT. Aspetta {wait_hint}, poi riprendi con:")
                print(f"  python3 push_to_hf.py --skip-big {len(big_files)} --skip-small-batches {b}")
                sys.exit(1)
            else:
                print(f"  ERRORE non rate-limit: {msg[:300]}. Salto questo batch.")
        time.sleep(2)

    print("\nUpload completato.")


if __name__ == "__main__":
    main()
