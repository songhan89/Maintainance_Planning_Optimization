## Problem Sets

There are three problem sets under ```A_set/```. 

```
├── A_set
|	├── A_01.json
|	├── A_06.json
|	├── A_10.json
```

## Codes

All files can be executed as **python filename.py --file problem_set_name**

```shell
python docplex_script.py --file A_01

python localbeamsearch.py --file A_06

python aco.py --file A_15
```

GA and SA use python package ```scikit-opt``` , so install the package before executing.
```
pip install scikit-opt
```
And run
```
python SA_skopt.py --file A_01

python GA.py --file A_06

python GA_multi.py --file A_15
```

## Output

Output files will be stored in ```output/``` with name of the algorithm and the problem set. The official checker for the competition is also in ```output/``` . The legitimacy of the outputs can be examined by

```
python output/RTE_ChallengeROADEF2020_checker.py A_set/A_15.json  output/A_15_ACO_60s.txt
```

