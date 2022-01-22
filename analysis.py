# -*- coding: utf-8 -*-
from numpy.random import seed
from numpy.random import randint
from numpy import mean
from numpy import median
from numpy import percentile
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from scipy import stats

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