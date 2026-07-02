"""
Verifica quale implementazione di check_correct viene risolta a runtime,
e se il confronto passa davvero da math_verify.

Uso:
    python debug_check_correct.py
"""
import inspect

from entropy.core.utils import check_correct

print("check_correct risolto da modulo:", check_correct.__module__)
print("file sorgente:", inspect.getsourcefile(check_correct))
print()
print("--- SOURCE ---")
print(inspect.getsource(check_correct))
print("--- END SOURCE ---\n")

# Test diretto sul caso Q0 reale
generated = (
    "...\n\\[\n21 + 49 = 70.\n\\]\n\n\\[\n\\boxed{70}\n\\]"
)
gt = "70"

result = check_correct(generated, gt)
print(f"check_correct(generated_with_boxed_70, '70') = {result}")

# Verifica se math_verify e' effettivamente importabile/usato
try:
    from math_verify import parse, verify
    print("math_verify import: OK ->", parse, verify)
except Exception as e:
    print("math_verify import FALLITO:", repr(e))
    print(">>> Se questo fallisce, verify_answer_equivalence NON puo' mai "
          "usare il path math_verify per casi non gia' risolti dal fast "
          "path 'normalized_exact' -- ma '70'=='70' dovrebbe comunque "
          "risolversi al fast path, quindi se anche questo test sopra da' "
          "False, il problema e' un check_correct diverso in risoluzione "
          "di import, non math_verify in se'.")