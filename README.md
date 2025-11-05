# Experiments for the out-of-core framework of SystemDS

## How to fetch experiments from remote

```sh
rsync -avz -e ssh --prune-empty-dirs \
  --include '*/' --include 'result.csv' --exclude '*' \
  so014:~/OOCExperiments/  ./
```