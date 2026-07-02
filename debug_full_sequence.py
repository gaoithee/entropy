"""
Diagnostica mirata sulla condizione `full_sequence` per Q0 (o altro indice),
da lanciare a mano nello stesso ambiente/venv del repo entropy.

Cosa stampa:
  1. Lunghezza e contenuto (decodificato) del prefix ricostruito
     (prompt + thinking[t_start:t_end] + end_ids) confrontato col
     prefix "grezzo" che il modello ha effettivamente visto in
     generazione originale (prompt + intero trace_tokens fino a
     end_thinking incluso, SENZA nessun re-inserimento di end_ids).
  2. Se i due prefix in token combaciano byte-per-byte fino al boundary.
  3. Quanti token vengono generati in greedy prima di:
       - emettere EOS, oppure
       - esaurire max_new_tokens
     (per capire se il greedy si sta troncando "in silenzio").
  4. Il testo generato per intero (non solo gli ultimi N caratteri),
     per vedere se un \boxed{} compare più avanti di quanto text_chars
     lasciasse intuire in gold_check.py.

Uso:
    python debug_full_sequence.py \
        --model openai/gpt-oss-20b \
        --traces_file data/aime2025/gpt-oss-20b_teacher_traces.json \
        --q_idx 0 \
        --max_new_tokens 400
"""
import json
from pathlib import Path

import torch
import fire

from entropy.core.data_utils import extract_boxed_answer
from entropy.models.registry import get_thinking_tokens

import evaluate_entropy_splice as ees


def main(
    model: str,
    traces_file: str,
    q_idx: int = 0,
    t_idx_hint: int = 0,
    max_new_tokens: int = 400,
    device: str = "cuda",
):
    path = Path(traces_file)
    with open(path) as f:
        questions = json.load(f)
    q = questions[q_idx]

    prompt_tokens = q["prompt_tokens"]
    traces_tokens = q["traces_tokens"]
    extracted_answers = q.get("extracted_answers", [None] * len(traces_tokens))

    lm, tokenizer, config = ees.load_lm(model)
    lm.eval()
    tokenizer = lm.tokenizer

    cfg = get_thinking_tokens(model)
    start_ids = cfg["start_token_ids"] or tokenizer.encode(cfg["start_token"], add_special_tokens=False)
    end_ids = cfg["end_token_ids"] or tokenizer.encode(cfg["end_token"], add_special_tokens=False)

    # trova la prima traccia utilizzabile, stesso criterio dello script principale
    t_idx = None
    for i, trace_tokens in enumerate(traces_tokens[t_idx_hint:], start=t_idx_hint):
        boundaries = ees._find_thinking_boundaries(trace_tokens, start_ids, end_ids)
        if boundaries is None:
            print(f"T{i}: nessun end_thinking, skip")
            continue
        orig_extracted = extracted_answers[i] if i < len(extracted_answers) else None
        if orig_extracted is not None and not str(orig_extracted).strip():
            print(f"T{i}: extracted_answers vuoto, skip")
            continue
        t_idx = i
        t_start, t_end = boundaries
        break

    if t_idx is None:
        print("Nessuna traccia utilizzabile trovata.")
        return

    trace_tokens = traces_tokens[t_idx]
    print(f"\n=== Q{q_idx} T{t_idx} ===")
    print(f"len(prompt_tokens) = {len(prompt_tokens)}")
    print(f"len(trace_tokens)  = {len(trace_tokens)}")
    print(f"boundaries (t_start, t_end) = {t_start, t_end}")
    print(f"thinking region length = {t_end - t_start}")

    # --- 1. Prefix ricostruito dallo script principale ---
    reconstructed = prompt_tokens + trace_tokens[t_start:t_end] + end_ids
    print(f"\nreconstructed prefix length = {len(reconstructed)}")
    print("last 20 tokens of thinking region (reconstructed, pre-end_ids):",
          trace_tokens[t_start:t_end][-20:])
    print("end_ids appended:", end_ids)
    print("decoded tail (last ~200 chars of reconstructed prefix):")
    print(repr(tokenizer.decode(reconstructed[-120:], skip_special_tokens=False)))

    # --- 2. Prefix "grezzo" come visto in generazione originale ---
    # NB: trace_tokens[t_end:] e' quello che il modello scrisse DOPO il suo
    # stesso end_thinking (cioe' l'inizio vero del canale final originale).
    original_full = prompt_tokens + trace_tokens
    print(f"\noriginal_full length = {len(original_full)}")
    print("what the model actually wrote right after its own end_thinking "
          "(next 40 tokens of trace_tokens[t_end:]):")
    tail_original = trace_tokens[t_end:t_end + 40]
    print(tail_original)
    print("decoded:", repr(tokenizer.decode(tail_original, skip_special_tokens=False)))

    # --- 3. Confronto: end_ids ricostruiti vs quello che il modello scrisse davvero ---
    real_end_marker = trace_tokens[t_end - 1: t_end - 1 + len(end_ids)]
    print(f"\nend_ids atteso (da registry):      {end_ids}")
    print(f"token reali nel trace a quel punto: {real_end_marker}")
    print("MATCH:", real_end_marker == end_ids)

    # --- 4. Generazione full_sequence, testo intero + diagnosi troncamento ---
    full_seq_ids = torch.tensor([reconstructed], device=device)
    gen_ids = ees.greedy_generate(
        lm, full_seq_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id
    )
    hit_eos = len(gen_ids) < max_new_tokens
    full_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print(f"\n=== GENERAZIONE full_sequence ===")
    print(f"n_tokens generated = {len(gen_ids)} / max_new_tokens = {max_new_tokens}")
    print(f"hit_eos_before_limit = {hit_eos}  (False => probabilmente TRONCATO)")
    boxed = extract_boxed_answer(full_gen_text)
    print(f"extract_boxed_answer(first=False/rfind) -> {boxed!r}")
    print(f"\nFULL generated text:\n{full_gen_text}")


if __name__ == "__main__":
    fire.Fire(main)