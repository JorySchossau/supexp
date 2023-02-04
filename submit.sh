
for i in {01..50}; do
  mkdir -p r${i}
  cp run.sbatch r${i}/
  cd r${i}
  sbatch run.sbatch &
  cd ..
done
