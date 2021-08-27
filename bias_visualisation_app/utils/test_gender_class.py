# This is the feature we are using to classify the names
# it is the last letter of the name

def gender_features(word):
  return{'Last letter': word[-1], 'First letter': word[0], 'Length': len(word)}

import nltk

# Here we will prepare a list of examples and corresponding class labels

from nltk.corpus import names
labeled_names = ([(names, 'male') for names in names.words('male.txt')] +
  [(names, 'female') for names in names.words('female.txt')])

import random
random.shuffle(labeled_names)

# Now we need to use the process the data with our feature and divide the resulting list
# into a training set and a test_api.py set; the training set is used to train a new
# 'naive Bayes' classifier

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Here we print the accuracy (low accuracy)
print(nltk.classify.accuracy(classifier, test_set))

# Let's check for the error we made when predicting name genders
# using a new set called dev-test_api.py set
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

train_set = [(gender_features(n), gender) for (n,gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n,gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess!= tag:
        errors.append((tag, guess, name))

for (tag, guess, name) in sorted(errors):
    print('correct={:<8} guess={:<8s} name:{:<30}'.format(tag, guess, name))

# We see that suffixes are important, because of that we will add this
# information to our feature extractor

def gender_features(word):
    return{'Suffix1': word[-1:],
    'Suffix2': word[-2:]}

train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))



# We can observe that the accuracy improved very much with this
# new feature extractor