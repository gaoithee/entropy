"""
Trova ed elimina i file eval finiti per errore nella root di uno o piu'
dataset su HF invece che dentro results/ o results_sentence/, cosi' li si
puo' ricaricare nel posto giusto senza duplicati.

Uso:
    python3 cleanup_misplaced.py <dataset1> [<dataset2> ...] [--dry-run]
    python3 cleanup_misplaced.py --all [--dry-run]   # tutti i dataset nel repo

Esempi:
    python3 cleanup_misplaced.py aime2025 --dry-run
    python3 cleanup_misplaced.py --all
"""
import sys
from huggingface_hub import HfApi

REPO_ID = "saracandu/entropy-traces"
REPO_TYPE = "dataset"

def find_misplaced(api, all_files, dataset):
    prefix = f"{dataset}/"
    to_delete = []
    for f in all_files:
        if not f.startswith(prefix):
            continue
        rest = f[len(prefix):]
        if "/" in rest:
            continue  # gia' dentro una sottocartella -> ok
        if rest.endswith("_teacher_traces.json"):
            continue  # file di root legittimo
        if "_eval_" in rest and rest.endswith(".json"):
            to_delete.append(f)
    return to_delete

def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    all_datasets_flag = "--all" in args
    positional = [a for a in args if not a.startswith("--")]

    api = HfApi()
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    if all_datasets_flag:
        datasets = sorted({f.split("/")[0] for f in all_files if "/" in f})
    elif positional:
        datasets = positional
    else:
        print("Uso: python3 cleanup_misplaced.py <dataset1> [<dataset2> ...] [--dry-run]")
        print("     python3 cleanup_misplaced.py --all [--dry-run]")
        sys.exit(1)

    per_dataset = {}
    total = 0
    for dataset in datasets:
        to_delete = find_misplaced(api, all_files, dataset)
        if to_delete:
            per_dataset[dataset] = to_delete
            total += len(to_delete)

    if total == 0:
        print("Nessun file mal posizionato trovato. Tutto pulito.")
        return

    print(f"Trovati {total} file eval mal posizionati su {len(per_dataset)} dataset:\n")
    for dataset, files in per_dataset.items():
        print(f"  {dataset}: {len(files)} file")
        for f in files[:3]:
            print(f"    {f}")
        if len(files) > 3:
            print(f"    ... e altri {len(files) - 3}")

    if dry_run:
        print("\n[DRY RUN] Nessuna eliminazione eseguita.")
        return

    confirm = input(f"\nEliminare questi {total} file dal repo remoto (tutti i dataset sopra)? [y/N] ")
    if confirm.lower() != "y":
        print("Annullato.")
        return

    from huggingface_hub import CommitOperationDelete
    all_to_delete = [f for files in per_dataset.values() for f in files]
    ops = [CommitOperationDelete(path_in_repo=f) for f in all_to_delete]
    batch_size = 200
    for i in range(0, len(ops), batch_size):
        chunk = ops[i:i+batch_size]
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=chunk,
            commit_message=f"Remove misplaced eval files from dataset roots ({i+1}-{i+len(chunk)}/{len(ops)})",
        )
        print(f"Eliminati {i+len(chunk)}/{len(ops)}")

    print("Pulizia completata.")

if __name__ == "__main__":
    main()
