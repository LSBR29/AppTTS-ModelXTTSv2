"""
Vuelca `vocab.json` (formato HuggingFace `tokenizers`, el mismo que usa VocesWindows/VocesLinux)
a tres archivos planos TSV que el tokenizador BPE en C++ de la app Android carga en tiempo de
ejecucion (`app/src/main/cpp/bpe_tokenizer.hpp`) -- no hace falta un parser JSON en C++.

No depende de PyTorch/TTS -- solo de `vocab.json`, así que corre con cualquier Python 3
(no hace falta el Python portable de VocesWindows para este paso en particular).

Uso:
    python dump_text_frontend.py \\
        --vocab ../../VocesWindows/models/2025acv02/vocab.json \\
        --outdir ../generated_models
"""
import argparse
import json
import os


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", default=os.path.join(here, "..", "..", "VocesWindows", "models", "2025acv02", "vocab.json"))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "generated_models"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.vocab, encoding="utf-8") as f:
        v = json.load(f)
    m = v["model"]
    vocab = m["vocab"]
    merges = m["merges"]
    assert m["type"] == "BPE", f"tipo de modelo de tokenizador inesperado: {m['type']!r}"
    assert v["pre_tokenizer"] == {"type": "Whitespace"}, "pre_tokenizer inesperado -- el parser C++ asume Whitespace"
    assert v["normalizer"] is None, "normalizer inesperado -- el parser C++ asume ninguno"

    for tok in vocab:
        assert "\t" not in tok and "\n" not in tok, f"token con delimitador TSV: {tok!r}"
    for pair in merges:
        assert "\t" not in pair, f"merge con delimitador TSV: {pair!r}"

    with open(os.path.join(args.outdir, "vocab.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write(f"{idx}\t{tok}\n")

    # El orden de las lineas ES el rank de prioridad de cada regla de fusion BPE -- no reordenar.
    with open(os.path.join(args.outdir, "merges.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for rank, pair in enumerate(merges):
            f.write(f"{rank}\t{pair}\n")

    added = v.get("added_tokens", [])
    with open(os.path.join(args.outdir, "special_tokens.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for tok in added:
            f.write(f"{tok['id']}\t{tok['content']}\n")

    print(f"vocab.tsv: {len(vocab)} tokens")
    print(f"merges.tsv: {len(merges)} reglas de fusion")
    print(f"special_tokens.tsv: {len(added)} tokens especiales")
    print("OK")


if __name__ == "__main__":
    main()
