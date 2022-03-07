## Readme

This code was started using the [code](https://github.com/SawanKumar28/nile) for NILE paper

## Running the code

Some of the instructions are common with the instructions from the original codebase.

1. Create a virtualenv and install dependecies
      ```bash
      virtualenv -p python3.6 env
      source env/bin/activate
      pip install -r requirements.txt
      ``` 

2. Datasets

The notebook control_codes.sh contains code to roughly identify the perturbation types in the CAD datasets collected for NLI and QA. The processed datasets for each perturbation type along with the dev and test sets are located in the data directory.

3. For any perturbation type for NLI (say lexical), run the following command to run experment for one random seed:

```
bash run_clf.sh 0 0 nli_pert_types instance instance _ train_paired_lexical dev_paired_lexical _ _ _ _
```

Similarly for QA, for any perturbation type (say lexical), run the following command:

```
bash run_qa.sh 0 0 qa_pert_types train_large_lexical dev_paired_lexical _
```

Note that in the paper, each experiment was run with random seeds (0 to 4), and the mean and std deviation were reported.

4. To test the above trained model on any perturbation type for SNLI (say test the above model on negation):

``` 
bash run_clf.sh 0 0 nli_pert_types instance instance _ _ test_paired_negation _ _ $PATH _
```

Similarly for QA

```
bash run_qa.sh 0 0 qa_pert_types _ test_paired_negation $PATH
``` 

where PATH is the path of the saved directory.
