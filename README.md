# painting-classification
Given a survey that asks to describe three different classes of famous paintings through different metrics (numerical, text, multiple choice), three different machine learning models were developed and trained to predict paintings for new data. Data was preprocessed and used by Logistic Regression, Naive Bayes, and Decision Trees for 91% accuracy.   

Presplit.py involves the preprocessing of a raw CSV file of student survey responses and fits the three different models of logistic regression, Naive Bayes, and decision trees. A split of 80/20 is made, and the accuracies for each model are compared, with logistic regression achieving the highest of 90.8%

Pre.py creates a JSON that includes all model artifacts (feature columns, weights, bias, etc.) for logistic regression.

Pred.py uses the JSON to develop a more sophisticated logistic regression model (using the same model artifacts) without the use of sklearn.

test.py ensures that pre.py and pred.py use the same model artifacts and  match entirely on target predictions.

<img width="368" height="271" alt="image" src="https://github.com/user-attachments/assets/70478786-0097-40ad-9d20-fabee443a6b6" />

<img width="368" height="271" alt="image" src="https://github.com/user-attachments/assets/14ecb618-4a85-41a9-bdbb-adb6da3d5c0b" />

<img width="368" height="271" alt="image" src="https://github.com/user-attachments/assets/c5cb8982-8699-400c-a2ad-4c31c5fe572d" />



