# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classification model trained to predict whether an individual earns more than $50K per year based on U.S. Census data. The model was trained using scikit-learn and deployed using FastAPI.

## Intended Use
The model is intended for educational and demonstration purposes only. It is designed to showcase how to build, evaluate, monitor, and deploy a machine learning model as part of an ML pipeline.

## Training Data
The model was trained on publicly available U.S. Census Bureau data containing demographic and employment-related features such as age, education, occupation, and hours worked per week.

## Evaluation Metrics
Model performance was evaluated using precision, recall, and F1 score. In addition, model performance was evaluated across categorical data slices to identify potential performance differences across groups.

## Ethical Considerations
The model may reflect historical biases present in the census data, particularly across sensitive attributes such as race and sex. Predictions should be interpreted with caution.

## Caveats and Recommendations
This model should not be used for real-world decision-making without additional validation, bias analysis, and ongoing monitoring. Further improvements could include bias mitigation and expanded evaluation across additional demographic slices.
