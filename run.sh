set -e

LOG=run_log.txt

{
  echo "==== GPU / CUDA environment ===="
  nvidia-smi || true
  nvcc --version

  echo
  echo "==== Clean build ===="
  rm -f attention_bench

  nvcc -O3 -std=c++17 -lineinfo \
    main.cu \
    attention_naive.cu \
    attention_tiled.cu \
    attention_online_tiled.cu \
    cpu_reference.cpp \
    -o attention_bench

  echo
  echo "==== Single sanity run ===="
  ./attention_bench --mode all --n 128 --d 64 --iters 100

  echo
  echo "==== Benchmark sweep ===="
  for n in 64 128 256 512; do
    for d in 32 64; do
      echo
      echo "---- n=${n}, d=${d} ----"
      ./attention_bench --mode naive  --n "$n" --d "$d" --iters 100
      ./attention_bench --mode tiled  --n "$n" --d "$d" --iters 100
      ./attention_bench --mode online --n "$n" --d "$d" --iters 100
    done
  done
} | tee "$LOG"

echo
echo "Saved log to $LOG"