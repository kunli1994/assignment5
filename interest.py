"""

Skeleton code. Note that this is not valid code due to all the dots.

"""

import random

import nltk
from nltk.corpus import senseval
from nltk.corpus.reader.senseval import SensevalInstance


def create_labeled_data():
    # collect all data from the corpus
    interest = senseval.instances('interest.pos')
    # create labeled data
    labeled_data = [(wsd_features(instance),instance.senses[0]) for instance in interest]
    print(len(labeled_data))
    return labeled_data


def create_feature_sets(labeled_data):
    # create feature sets



    test_set, train_set = labeled_data[round(len(labeled_data) * 0.9):], labeled_data[:round(len(labeled_data) * 0.9)]
    return train_set, test_set


def wsd_features(instance):
    # pre = {'pre':instance.context[instance.position - 1][0]}
    # pos = {'pos':instance.context[instance.position + 1][0]}
    # pre_tag = {'pre_tag':instance.context[instance.position - 1][1]}
    # pos_tag = {'pos_tag':instance.context[instance.position + 1][1]}



    return {'pre':instance.context[instance.position - 1][0],'pos':instance.context[instance.position + 1][0],'pre_tag':instance.context[instance.position - 1][1],'pos_tag':instance.context[instance.position + 1][1]}


def make_instance(tagged_sentence):
    words = [t[0] for t in tagged_sentence]
    position = words.index('interest')
    return SensevalInstance('interest-n', position, tagged_sentence, [])


def train_classifier(training_set):
    # create the classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier



def evaluate_classifier(classifier, test_set):
    # get the accuracy and print it
    print("Accuracy: ",nltk.classify.accuracy(classifier,test_set))



def run_classifier(classifier):
    emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
    sents = [s for s in emma if 'interest' in s]
    tag_sents = [nltk.pos_tag(sent) for sent in sents]
    instances = [make_instance(tag_sent) for tag_sent in tag_sents]
    for ins in instances:
        ins.senses = classifier.classify(wsd_features(ins))
        lst = [word for (word,tag) in ins.context]
        lst[ins.position] = ins.senses
        print(' '.join(lst))





if __name__ == '__main__':

    labeled_data = create_labeled_data()
    training_set, test_set = create_feature_sets(labeled_data)
    classifier = train_classifier(training_set)
    evaluate_classifier(classifier, test_set)
    run_classifier(classifier)
