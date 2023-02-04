for i in {00..50}; do grep @ r${i}/data.off | cut -d'@' -f2 > r${i}/off.csv; grep @ r${i}/data.on | cut -d'@' -f2 > r${i}/on.csv; done
