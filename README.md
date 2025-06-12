# Gradient Boosting versus Mixed Integer Programming for Sparse Additive Modeling

MIPRule is a Python-based framework for generating additive rule ensembles using Mixed Integer Programming (MIP) approaches. The project implements various optimization techniques for learning interpretable rule-based models that can be used for both classification and regression tasks.

## Prerequisites

This project requires Gurobi optimization solver. You must have a valid Gurobi license installed on your computer. You can obtain a license from [Gurobi's website](https://www.gurobi.com/downloads/).

## Project Structure

The project is organized into several key directories:

- `optimization/`: Core implementation of rule learning algorithms
  - `boosting_col2.py`: Implementation of the boosting algorithm
  - `fc_boosting_col2.py`: Implementation of the fully-corrective boosting algorithm
  - `opt_fc_boosting2_col.py`: Implementation of the MIP-based rule generation algorithm
  - Supporting modules for rule learning and optimization

## Key Features

1. **Multiple Loss Functions**:
   - Squared Loss (regression)
   - Logistic Loss (classification)
   - Poisson Loss (count data)

2. **Rule Learning Approaches**:
   - Idealized Boosting and Fully-corrective Boosting with Rules implemented with MIP
   - Additive rule ensembles with MIP optimization

3. **Model Types**:
   - Additive Rule Ensembles

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MIPRule.git
cd MIPRule

# Install dependencies
pip install -r requirements.txt
```

<!-- ## Usage

### Boosting Algorithm

```python
from optimization.boosting_col2 import rule_boosting2

# Run boosting algorithm
ensemble, risk, bnd = rule_boosting2(
                n, d, k, L, U, train, train_target, labels, 
                loss_func='squared', tl=600, f=None, reg=0.1, debug=False,
                max_col_num=10)
```

### MIP-based Rule Generation

```python
from optimization.opt_fc_boosting2_col import optimized_rule_ensemble2

# Run MIP-based rule generation
risk, ensembles, bnd = fc_opt_boosting(n, d, k, L, U, train, train_target, labels,
                                      loss_func='squared', tl=600, reg=0.1, debug=False,
                                      f=None, max_col_num=10)
```
->

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included in the repository.

## Citation

If you use this software in your research, please cite:

```
@software{miprule2025,
  author = {Fan Yang},
  title = {MIPRule: Mixed Integer Programming for Rule Ensemble Generation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/MIPRule}
}
```