---
title: "Machine Learning Model Selection and Validation"
teaching: 30
exercises: 30
questions:
- "Key question (FIXME)"
objectives:
- "First learning objective. (FIXME)"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

In the last session we trained a classification model on some micrscope imagery conatin protein crystals. Towards the end of the session we applied a range of different performance metrics that illuminated different aspects of the models performance. Ultimately we observed that the model was heavily biased by the unbalanced distribution of classes in our training set. How do we know this wasn't just by chance? Can we get some statistical guarantee about our findings?

## Single Model Validation

A nice way to test statistical significance is through the use of [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics "Cross-Validation Wikipedia page"). For each fold of validation we can measure an accuracy (or some other metric of interest), thus with a sample of accuracies we can now estimate a distribution over accuracies. With a distribution of accuracies we can now apply a [Student's t-Test](https://en.wikipedia.org/wiki/Student%27s_t-test "Wikipedia t-Test page") to test for a statistically significant difference from some benchmark value. We can do it all with a few lines of Python code. In case your environment has reset, or you are coming in to this session raw, let's just redownload or data and define our classifier.

~~~
from utils.datasets import c3
(X_train, y_train), (X_test, y_test) = c3.load_data()
~~~
{: .language-python}

~~~
Downloading datafile to /root/data/crystals.npy ...
... done

Downloading datafile to /root/data/clear.npy ...
... done

Sub-sampling dataset...
... shuffling and splitting
... done
~~~
{: .output}
> ## Authentication
>
>The first time you execute one of the dataset utility functions you will be asked to authenticate with you google credentials, don't be afriad.
>
> ~~~
> Go to the following link in your browser:
>
>    https://accounts.google.com/o/oauth2/auth?redirect_uri=.....
>
>Enter verification code:
> ~~~
> {: .output}
{: .callout}
~~~
from sklearn import svm
classifier = svm.SVC(gamma=0.001, verbose=True)
~~~
{: .language-python}
~~~
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=True)
~~~
{: .output}

## Cross Validation

Cross validation can be implemented using `scikit-learn` in a single line

~~~
from sklearn.model_selection import cross_validate

cv_results = cross_validate(classifier, X_train, y=y_train, cv=10, verbose=3, n_jobs=-1)
~~~
{: .language-python}
~~~
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:  5.3min finished
~~~
{: .output}

The above code will perform a 10-fold cross-validation on our traing data set. You can specify specific metrics that you would like to collect across the folds, but there are some default values chosen for you 

~~~
print(cv_results.keys())
~~~
{: .language-python}
~~~
dict_keys(['fit_time', 'score_time', 'test_score', 'train_score'])
~~~
{: .output}
For our t-Test we are interested in the `test_score` metric, which in this case is simply the accuracy. We can now calculate the one-sided t-statistic for our accuracy and some benchmark value, lets say an accuracy of 80%.

~~~
from scipy import stats

accuracies = cv_results["test_score"]
k = len(accuracies)
benchmark = 0.8

m = np.mean(accuracies)
S = np.std(accuracies)
SE = S / np.sqrt(k)

t = (m - benchmark) / SE

p = stats.t.cdf(t,df=2*k - 2)
print("The accuracy of you model is %.2f\u00B1%.2f" %(m, 2.58*SE))
print("The p-value for this test is %.3f" % p)
~~~
{: .language-python}
~~~
The accuracy of you model is 0.80±0.00
The p-value for this test is 0.103
~~~
{: .output}

In this case it looks like we cannot say with confidence that our model accuracy is greater than or equal to 80%.

{% include links.md %}

