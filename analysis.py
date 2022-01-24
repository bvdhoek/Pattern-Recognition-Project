# -*- coding: utf-8 -*-
from numpy.random import seed
from numpy.random import randint
from numpy import mean
from numpy import median
from numpy import percentile
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from scipy import stats
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
    
    #With the test values, the table should be interpreted as: the p-value for difference between model 1 and model 2 is 0.0158 meaning that there
    #is a statistically significant difference between model 1 and 2. Same goes for models 2 and 3, as their alpha is 0.04.
    #The p value for the comparison between model 1 and 3 is higher than 0.05 meaning that there is no significant difference between model 1 and 3
    
#test values:    
test_model_1 = [85, 86, 88, 75, 78, 94, 98, 79, 71, 80]
test_model_2 = [91, 92, 93, 90, 97, 94, 82, 88, 95, 96]
test_model_3 = [79, 78, 88, 94, 92, 85, 83, 85, 82, 81]

ANOVA_test(test_model_1, test_model_2, test_model_3)