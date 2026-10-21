# Folder Structure

- Analysis: Code used for analysing the results, such as for printing them to txt files or generating plots.
- Experiments: Code containing the tests. It contains three types of tests: the sanity checks, the preliminary tests, and the full tests (tests for which the results were analysed in the report)
- Plots: All the plots that were generated to analyse the results
- Results: Results of the preliminary tests and full tests
- Src: Main code containing all the coresets, testing framework, and dataset preperation



# Running The code:


### Setting up environment with conda:
```bash
conda create -n grid-coreset python=3.12.9 -y
conda activate grid-coreset
pip install -r requirements.txt
pip install -e .
```

### Running Experiments:
```bash
python experiments/test_folder/test_file.py
```
replace test_folder with the folder of the test, and test_file.py with the filename of your test.

example experiment command:
```bash
python experiments/sanity_checks/test_grid_mean.py
```

