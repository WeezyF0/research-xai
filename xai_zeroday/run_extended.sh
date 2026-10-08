#!/bin/bash
# Extended pilot: NSL-KDD official + UNSW-NB15 leave-one-family-out (DoS, Exploits, Reconnaissance), 3 seeds each.
cd "$(dirname "$0")"
for seed in 0 1 2; do
  for cfg in "nsl_kdd official -" "unsw_nb15 lofo DoS" "unsw_nb15 lofo Exploits" "unsw_nb15 lofo Reconnaissance"; do
    set -- $cfg
    out="results/extended/$1__$2__$3__seed$seed"
    [ -f "$out/report.md" ] && continue
    fam=""; [ "$3" != "-" ] && fam="--family $3"
    echo "[$(date +%T)] $out"
    .venv/bin/python -u quick_study.py --dataset $1 --protocol $2 $fam --seed $seed --max-fit 50000 --out "$out" > "$out.log" 2>&1 || echo "FAILED $out"
  done
done
echo ALL_DONE
