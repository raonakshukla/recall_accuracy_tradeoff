#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,precision_recall_fscore_support,accuracy_score,confusion_matrix, ConfusionMatrixDisplay


# Function for optimizing the recall for both the classes
def recall_tradeoff(pred_prob,y_test):
    
    """Return the predicted values for a threshold that optimizes recall for both classes.
     Input:
        pred_prob- An array of predicted probabilities by the given model. It should be(nx2)
        y_test- Target variable on test data. It should be binary
        
    Output:
        y_pred- Predicted label for optimum probability threshold
        lowest_prob- Probability threshold calculated by recall_tradeoff
        Plots the probability threshold where recall for both the classes are equal"""
    
    shape = pred_prob.shape[1]
    
    if shape > 2:
        print("Sorry! the module is designed for binary classification problem only")
        
    elif shape == 2:
        preds_df = pd.DataFrame(pred_prob[:,1], columns = ['prob_default'])
        thresh=[]
        default=[]
        non_defaut=[]
        accs= []
        for i in np.arange(0,1,0.01):
            thresh.append(i)
            preds_df['target'] = preds_df['prob_default'].apply(lambda x: 1 if x > i else 0)
            accuracy = accuracy_score(y_test,preds_df['target'])
            accs.append(accuracy)
            default_recall = precision_recall_fscore_support(y_test,preds_df['target'])[1][1]
            default.append(default_recall)
            non_default_recall = precision_recall_fscore_support(y_test,preds_df['target'])[1][0]
            non_defaut.append(non_default_recall)
    
    else:
        preds_df = pd.DataFrame(pred_prob, columns = ['prob_default'])
        thresh=[]
        default=[]
        non_defaut=[]
        accs= []
        for i in np.arange(0,1,0.01):
            thresh.append(i)
            preds_df['target'] = preds_df['prob_default'].apply(lambda x: 1 if x > i else 0)
            accuracy = accuracy_score(y_test,preds_df['target'])
            accs.append(accuracy)
            default_recall = precision_recall_fscore_support(y_test,preds_df['target'])[1][1]
            default.append(default_recall)
            non_default_recall = precision_recall_fscore_support(y_test,preds_df['target'])[1][0]
            non_defaut.append(non_default_recall)
    
    diff = np.array(default) - np.array(non_defaut)
    positive_difference = [d for d in diff if d > 0]
    lowest_prob = thresh[np.argmin(positive_difference)]
    y_pred = preds_df['prob_default'].apply(lambda x: 1 if x > lowest_prob else 0)
    plt.plot(thresh,default)
    plt.plot(thresh,non_defaut)
    plt.plot(thresh,accs)
    plt.axvline(x = lowest_prob, color = 'b', label = 'Threshold')
    plt.xlabel("Probability Threshold")
    plt.legend(["Default Recall","Non-default Recall","Model Accuracy","Threshold"])
    plt.show()
    print(f"The probability Threshold is {lowest_prob}")
    return y_pred,lowest_prob



# Function to visualize Confusion Matrix based on optimum probability threshold returned by "recall_tradeoff"
def confusion_matrix_opt(pred_prob,y_test,lowest_prob):
    
    """Plots the Confusion Matrix for optimum probability threshold
    Input:
        pred_prob- An array of predicted probabilities by the given model
        y_test- Target variable on test data. It should be binary
        lowest_prob- Probability threshold calculated by recall_tradeoff
    Output:
        Confusion Matrix for optimum probability threshold"""
    
    shape = pred_prob.shape[1]
    
    if shape > 2:
        print("Sorry! the module is designed for binary classification problem only")
        
    elif shape == 2:
        preds_df = pd.DataFrame(pred_prob[:,1], columns = ['prob_default'])
        y_pred_opt = preds_df['prob_default'].apply(lambda x: 1 if x > lowest_prob else 0)               
        cm_opt = confusion_matrix(y_test, y_pred_opt)
        disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt)
    else:
        preds_df = pd.DataFrame(pred_prob, columns = ['prob_default'])
        y_pred_opt = preds_df['prob_default'].apply(lambda x: 1 if x > lowest_prob else 0)
        cm_opt = confusion_matrix(y_test, y_pred_opt)
        disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt)
    print("The Confusion Matrix for Optimised Threshold:\n")
    disp_opt.plot()
    plt.show()


# Function to visualize Classification Report based on predictions made for threshold probability by "recall_tradeoff"
def classification_report_opt(y_test,y_pred):
    
    """Classification Report for optimum probability threshold
    Input:
        y_test- Target variable on test data. It should be binary
        y_pred- Predicted label for optimum probability threshold
    Output:
        Classification Report for optimum probability threshold"""
    
    print("The classification report for optimum threshold:\n",classification_report(y_test,y_pred))
    



               













        
        
        
    