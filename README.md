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
   - Gradient Boosting with Rules
   - Mixed Integer Programming optimization
   - Column Generation techniques

3. **Model Types**:
   - Additive Rule Ensembles
   - Rule-based boosting
   - XGBoost-inspired rule learning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MIPRule.git
cd MIPRule

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Boosting Algorithm

```python
from optimization.boosting_col2 import boosting_step2

# Run boosting algorithm
model = boosting_step2(
    X_train,
    y_train,
    loss='squared',  # or 'logistic' for classification
    num_rules=10
)
```

### Fully-Corrective Boosting Algorithm

```python
from optimization.fc_boosting_col2 import fully_corrective2

# Run fully-corrective boosting algorithm
model = fully_corrective2(
    X_train,
    y_train,
    loss='squared',  # or 'logistic' for classification
    num_rules=10
)
```

### MIP-based Rule Generation

```python
from optimization.opt_fc_boosting2_col import optimized_rule_ensemble2

# Run MIP-based rule generation
model = optimized_rule_ensemble2(
    X_train,
    y_train,
    loss='squared',  # or 'logistic' for classification
    num_rules=10
)
```

## Evaluation

The framework includes comprehensive evaluation tools:

```python
from evaluation.evaluate import evaluate_model

# Evaluate model performance
results = evaluate_model(
    model,
    X_test,
    y_test,
    metrics=['accuracy', 'auc']
)
```

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