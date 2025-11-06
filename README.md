# Experiments for the out-of-core framework of SystemDS

## How to fetch experiments from remote

```sh
rsync -avz -e ssh --prune-empty-dirs \
  --include '*/' --include 'results.csv' --exclude '*' \
  so014:~/OOCExperiments/  ./
```

## How to update SystemDS jar

```sh
scp SystemDS.jar so014:~/lib/
```