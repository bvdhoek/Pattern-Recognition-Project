# -*- coding: utf-8 -*-
from numpy.random import seed
from numpy.random import randint
from numpy import mean
from numpy import median
from numpy import percentile
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from scipy import stats
import evaluation
from sklearn.metrics import top_k_accuracy_score
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
#code from https://machinelearningmastery.com/confidence-intervals-for-machine-learning/

#Monte Carlo method (I think)

#as far as I understand it, we get the model to make predictions based on random 
#samples of the data and calculate (95%) confidence interval. Can see if there is an 
#overlap. If a model outperforms another on classes inside this overlap, this may be
# due to chance (I think???)
def calculate_percentile(model, data, alpha):
    # seed the random number generator
    seed(1)
    # bootstrap
    scores = list()
    for _ in range(100):
    	# bootstrap sample
    	indices = randint(0, 1000, 1000)
    	sample = data[indices]
    	# calculate and store statistic
    	statistic = mean(model.predict(sample))
    	scores.append(statistic)
    print('50th percentile (median) = %.3f' % median(scores))
    # calculate 95% confidence intervals (100 - alpha) if alpha == 5
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = max(0.0, percentile(scores, lower_p))
    print('%.1fth percentile = %.3f' % (lower_p, lower))
    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = min(1.0, percentile(scores, upper_p))
    print('%.1fth percentile = %.3f' % (upper_p, upper))


#mcnemar test if we feel like it. May be useful to see which predictions are done better
#by different models
def mcnemar_test(model1, model2, testX, testY):
    model1_predictions = model1.predict(testX)
    model2_predictions = model2.predict(testX)
    comparison_table = mcnemar_table(y_target=testY, y_model1= model1_predictions, y_model2= model2_predictions)
    print(comparison_table)
    
    chi2, p = mcnemar(ary=comparison_table)
    print('chi-squared:', chi2)
    print('p-value:', p)
    
#Can also do t-tests for each of the 50 classes to see where the difference is 
#significant

#1 or 2 tailed?? I think 2 tailed

def t_test(model1, model2, data):
    predictions1 = model1.predict(data)
    predictions2 = model2.predict(data)
    
    for i in range(50):
        print('Comparison of models for class ' + str(i))
        print()
        stats.ttest_ind(predictions1[i], predictions2[i],
                        equal_var = False, alternative = 'two sided')
        
def ANOVA_test(model_1_predictions, model_2_predictions, model_3_predictions):
    
    print(f_oneway(model_1_predictions, model_2_predictions, model_3_predictions))
    
    #With the test values, we can see an alpha value of less than 0.05, this means that the null hypothesis (the means are the same)
    # can be rejected meaning there is a statistical difference between the models. Now we do the Tukeys test to determine which groups are different
    
    
    concatenated = model_1_predictions + model_2_predictions + model_3_predictions
    
    df = pd.DataFrame({'predictions': concatenated, 'Model': np.repeat(['model_1', 'model_2', 'model_3'], repeats=len(model_1_predictions))}) 
   
    tukey = pairwise_tukeyhsd(endog=df['predictions'], groups=df['Model'], alpha=0.05)
    print(tukey)
    
    
def removesuffix(string):
    if string.endswith(' km'):
        return string[:-3]
    

        
labels = pd.read_csv('2K_states_int.csv')
#labels_str = pd.read_csv('2K_states_str.csv')

probs_mlp = pd.read_csv('MLP_and_ResNet/MlpModel_100imgs_25epochs_15_predictions.csv')
probs_resnet = pd.read_csv('MLP_and_ResNet/ResNetModel_100imgs_25epochs_13_predictions.csv')
probs_mixed = pd.read_csv('CombinedModelEnd_100imgs_25epochs_predictions.csv')

probs_mlp = probs_mlp.iloc[:,1:51]
probs_resnet = probs_resnet.iloc[:,1:51]
probs_mixed = probs_mixed.iloc[:,1:51]

preds_mlp = evaluation.make_preds(probs_mlp)
preds_resnet = evaluation.make_preds(probs_resnet)
preds_mixed = evaluation.make_preds(probs_mixed)


#dist_mlp = evaluation.evaluate_classification(preds_mlp, labels)
#dist_resnet = evaluation.evaluate_classification(preds_resnet, labels)
#dist_mixed = evaluation.evaluate_classification(preds_mixed, labels)

def top_k_acc():
    for k in (1,2,3,5):
        print('top ' + str(k) + 'accuracy for MLP: ' + str(top_k_accuracy_score(labels, probs_mlp, k = k)))
        print('top ' + str(k) + 'accuracy for ResNet: ' + str(top_k_accuracy_score(labels, probs_resnet, k = k)))
        print('top ' + str(k) + 'accuracy for Mixed model: ' + str(top_k_accuracy_score(labels, probs_mixed, k = k)))
        print()
    
def get_distances():
    distances_mlp = np.zeros((len(labels),1))
    distances_resnet = np.zeros((len(labels),1))
    distances_mixed = np.zeros((len(labels),1))
    for i in range(len(labels)):
        distances_mlp[i] = float(removesuffix(str(evaluation.evaluate_classification(probs_mlp.iloc[i,:], int(labels.iloc[i,:])))))
        distances_resnet[i] = float(removesuffix(str(evaluation.evaluate_classification(probs_resnet.iloc[i,:], int(labels.iloc[i,:])))))
        distances_mixed[i] = float(removesuffix(str(evaluation.evaluate_classification(probs_mixed.iloc[i,:], int(labels.iloc[i,:])))))
    
    return distances_mlp, distances_resnet, distances_mixed

def get_correct_preds():
    corr_preds_mlp = []
    corr_preds_resnet = []
    corr_preds_mixed = []
    for i in range(len(labels)):
        
        label = int(labels.iloc[i])
        
        if preds_mlp[i] == label:
            corr_preds_mlp.append(label)
            
        if preds_resnet[i] == label:
            corr_preds_resnet.append(label)   
            
        if preds_mixed[i] == label:
            corr_preds_mixed.append(label)
    return corr_preds_mlp, corr_preds_resnet, corr_preds_mixed

def plot_hist(data):
    pd.DataFrame(data).hist()
        
#ANOVA_test(distances_mlp.reshape(100013,), distances_resnet.reshape(100013,), distances_mixed.reshape(100013,))
#test values:    
#test_model_1 = [85, 86, 88, 75, 78, 94, 98, 79, 71, 80]
#test_model_2 = [91, 92, 93, 90, 97, 94, 82, 88, 95, 96]
#test_model_3 = [79, 78, 88, 94, 92, 85, 83, 85, 82, 81]