
import csv
import random, math
import numpy as np


#Load the CSV file with data aranged such that the attributes appear in columns with the last 
#column as the class in either a 0 or 1
def loadCsvfile(filepath):
    lines = csv.reader(open(filepath,"rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

#Split the dataset into a training and test set according to a split ratio
def splitData(dataset, splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet = []
    copy = dataset[:]
    while len(trainSet)<trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    print("Split {0} cases into {1} Training and {2} Test").format(len(dataset),len(trainSet),len(copy))
    return[trainSet,copy]

def seperateClass(dataset):
    seperate = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in seperate:
            seperate[vector[-1]] = []
        seperate[vector[-1]].append(vector)
    return seperate

def summarize(dataset):
    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    seperatedData = seperateClass(dataset)
    summaries = {}
    for classValue, instances in seperatedData.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def PDF(x,mean,std):
    return (1 / (math.sqrt(2*math.pi)*std)) * math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))

def ClassProb(summaries,inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] += math.log(PDF(x,mean,std))
    return probabilities

def predict(summaries, inputVector):
    probabilities = ClassProb(summaries, inputVector)
    bestLabel,bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel == None or probability>bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPred(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)

    return predictions

def getAccuracy(predictions,testSet):
    accuracy = 0.0
    for i in range(len(predictions)):
        vector = testSet[i]
        if predictions[i] == vector[-1]:
            accuracy += 1
    return (accuracy/float(len(predictions)))*100.0


def main():
    filepath = ""  #Insert file path here
    data = loadCsvfile(filepath)
    traindata, testdata = splitData(data, 0.8) #the ratio here determines the train:test split ratio
    summary = summarizeByClass(traindata)  #sumary contains the log(P(spatiotemporal gait variable | Class)) for all variables over all classes
    predictions = getPred(summary, testdata)  #We add the log(P) over all attributes and look at the largest one to determine class
    accuracy = getAccuracy(predictions, testdata)  
    print('Your naive Gaussian Bayes got an accuracy of: {0}').format(accuracy)


main()