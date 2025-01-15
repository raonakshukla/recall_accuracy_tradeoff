## recall_optimiser: A package for optimizing recall by finding the best probability threshold for binary classification
The classification of an instance into either of the classes depends on the probability threshold. Usually for binary classification the threshold used by any classification algorithm is 50% which makes sense as there are only two classes in the classification problem. However, a 50% probability may not be the optimum threshold and can lead to loss in critical cases like medical or financial cases. For example, if a cancerous patient is classified as non-cancerous by an algorithm it may lead to misdiagnosis or in the worst case can lead to the death of the patient. Similarly in the loan default prediction problem if a person who defaults is misclassified as non-default it will lead to a huge financial loss to the lender. 

The module aims to simulate various probability thresholds between 0 and 1 and calculate recall for both classes. For a particular threshold, the recall for both classes will be equal thereby minimizing the loss. In the loan default prediction problem misclassification of a defaulter as non-default leads to financial loss to the exchequer. However, if a non-default is classified as default it leads to rejection of loan application and loss of opportunity cost. The solution to the problem is to find a threshold where recall for both cases is equal thereby minimizing the loss on either end. 


## Functions in the module
* "recall_tradeoff" is designed to simulate the process and find the optimum threshold probability.
  * Input
    * pred_prob- An array of predicted probabilities by the given model(nx2)
    * y_test- Target variable on test data. It should be binary
  * Output
    * y_pred- Predicted label for optimum probability threshold
    * lowest_prob- Probability threshold calculated by recall_tradeoff
    * Plots the probability threshold where recall for both the classes are equal
      
* "confusion_matrix_opt" plots the confusion matrix for threshold probability.
  * Input
    * pred_prob- An array of predicted probabilities by the given model
    * y_test- Target variable on test data. It should be binary
    * lowest_prob- Probability threshold calculated by recall_tradeoff
  * Output
    * Confusion Matrix for optimum probability threshold
      
* "classification_report_opt" generates a classification report for threshold probability.
  * Input
    * y_test- Target variable on test data. It should be binary
    * y_pred- Predicted label for optimum probability threshold
  * Output
    * Classification Report for optimum probability threshold

## Please cite this paper if you are using recall_optimiser:
To be submitted
DOI: To be submitted

## References
This toolbox was developed by Raonak Shukla a master's student as a part of his dissertation on credit risk modeling from [Department of Computer Science, University of Nottingham](https://www.nottingham.ac.uk/computerscience/about/about-us.aspx) supervised by [Prof. Rong Qu](https://people.cs.nott.ac.uk/pszrq/index.html). 

 
