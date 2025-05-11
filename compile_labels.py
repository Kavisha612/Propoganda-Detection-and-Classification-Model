#!/usr/bin/env python3
import os, glob
import argparse

def compile_span(input_dir, output_path):
    """
    Read all *.task1‑SI.labels in input_dir and write a single
    tab‑separated file with header: doc_id, start, end
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.labels")))
    with open(output_path, "w", encoding="utf‑8") as out:
        # out.write("doc_id\tstart\tend\n")
        for fn in files:
            with open(fn, encoding="utf‑8") as f:
                for line in f:
                    line = line.strip()
                    # skip header (the first line is the file name)
                    if not line or line.startswith("article"):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        doc_id, start, end = parts[0], parts[1], parts[2]
                        out.write(f"{doc_id}\t{start}\t{end}\n")

def compile_tc(input_dir, output_path):
    """
    Read all *.task2‑TC.labels in input_dir and write a single
    tab‑separated file with header: doc_id, start, end, technique
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.labels")))
    with open(output_path, "w", encoding="utf‑8") as out:
        # out.write("doc_id\tstart\tend\ttechnique\n")
        for fn in files:
            with open(fn, encoding="utf‑8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("article"):
                        continue
                    parts = line.split()
                    # parts = [doc_id, technique, start, end]
                    if len(parts) == 4:
                        doc_id, technique, start, end = parts
                        out.write(f"{doc_id}\t{start}\t{end}\t{technique}\n")
                    else:
                        # in case technique contains spaces (unlikely), re‑join
                        doc_id = parts[0]
                        start = parts[-2]
                        end = parts[-1]
                        technique = " ".join(parts[1:-2])
                        out.write(f"{doc_id}\t{start}\t{end}\t{technique}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compile pilot span & technique labels into two flat files")
    p.add_argument("--span-dir",   required=True, help="pilot_train-labels-task1-span-identification folder")
    p.add_argument("--tech-dir",   required=True, help="pilot_train-labels-task2-technique-classification folder")
    p.add_argument("--out-span",   default="pilot_train-task1-SI.labels", help="output file for spans")
    p.add_argument("--out-tech",   default="pilot_train-task2-TC.labels", help="output file for techniques")
    args = p.parse_args()

    compile_span(args.span_dir, args.out_span)
    compile_tc(args.tech_dir, args.out_tech)

    print(f"Wrote spans → {args.out_span}")
    print(f"Wrote techniques → {args.out_tech}")
