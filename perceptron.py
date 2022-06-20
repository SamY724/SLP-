import numpy as np
import pandas as pd
import random

#reading the file's using pandas
train = pd.read_csv("train.data.csv",header=None)

test = pd.read_csv("test.data.csv",header=None)


#the data file's are then casted to a numpy array to allow me to manipulate it

data_Array = train.to_numpy()


test_Array = test.to_numpy()


#Weights for the binary 1v1 classifiers:
weights1 = [0,0,0,0,0]
weights2 = [0,0,0,0,0]
weights3 = [0,0,0,0,0]

#Weights for the 1 vs rest classifiers:
weights1vAll = [0,0,0,0,0]
weights2vAll = [0,0,0,0,0]
weights3vAll = [0,0,0,0,0]



def perceptronTrain(trainData,iterCount):

    #Initialising weights, whilst there are only four feature values, an additional 0 is inserted at index 0 to pre-establish the bias term

   
    global weights1

    weights1 =[0,0,0,0,0]

    global weights2

    weights2 = [0,0,0,0,0]

    global weights3

    weights3 = [0,0,0,0,0]

    global weights1vAll

    weights1vAll = [0,0,0,0,0]

    global weights2vAll

    weights2vAll = [0,0,0,0,0]

    global weights3vAll

    weights3vAll = [0,0,0,0,0]


    # Setting value for random seed, in order to make random shuffling consistent and not create fluctuating values for weights and bias        
    
    np.random.seed(1)


    # for the 1vs1 approach of q3, the datasets will need to be cleaned to the two classes, which is done through manipulating my training array below


    classifier1 = np.delete(trainData,np.where(trainData == "class-3")[0],axis=0)

        
    classifier2 = np.delete(trainData,np.where(trainData == "class-1")[0],axis=0)


    classifier3 = np.delete(trainData,np.where(trainData == "class-2")[0],axis=0)



    for iter in range(1,iterCount):



        # I establish my correct class counts within the iteration loop, this is done in order to refresh the correct classification count upon each iter
        # This means I have easy access to seeing exactly how many accurate predictions are made at the final epoch

        correct_class_count1 = 0

        correct_class_count2 = 0

        correct_class_count3 = 0

        correct_class_count1vall = 0

        correct_class_count2vall = 0

        correct_class_count3vall = 0

        correct_prediction = 0

    
        # In order to adhere to best training practice for my perceptron algorithm, the classifiers are shuffled within in the iteration loop
        # with the random seed established earlier, the shuffling is consistent and does not fulctuate after each iteration
        # which would cause inconsistent weights and bias terms upon each iteration

        np.random.shuffle(classifier1)

        np.random.shuffle(classifier2)

        np.random.shuffle(classifier3)

        np.random.shuffle(trainData)
       

        for x in classifier1:

            # x is casted as a list to allow me ease access to insert 1 at index 0, in order to counteract the pre-inserted bias term i have in my weightvector
            # this is done in order to conform to the notation trick shown in the lectures and saves me creating multiple bias variables for my classifiers

            x = list(x)
            x.insert(0,1)

            a = np.dot(weights1,x[:-1])

            
            if x[-1] == "class-1":
                y = 1
            else:
                y = -1


            if y*a <= 0:
                    weights1[0] += y
                    weights1[1] += y*x[1]
                    weights1[2] += y*x[2]
                    weights1[3] += y*x[3]
                    weights1[4] += y*x[4]
            else:
                    correct_class_count1 += 1


        # The below feature puts my correct class count to use, which will calculate the amount of miscalculations at the 20th iteration
        # As such, I take the number of objects using .shape to provide my "whole object set" count, and divide this by the correct classifications yielded
        # From the activation score, this is repeated for the other classifiers also.

        objects1,features1 = classifier1.shape

        accuracy1 = correct_class_count1/objects1




        for x in classifier2:


            x = list(x)
            x.insert(0,1)

            a = np.dot(weights2,x[:-1])

            
            if x[-1] == "class-2":
                y = 1
            else:
                y = -1


            if y*a <= 0:
                weights2[0] += y
                weights2[1] += y*x[1]
                weights2[2] += y*x[2]
                weights2[3] += y*x[3]
                weights2[4] += y*x[4]
            else:
                correct_class_count2 += 1


        objects2,features2 = classifier2.shape

        accuracy2 = correct_class_count2/objects2

        


        for x in classifier3:

            x = list(x)
            x.insert(0,1)

            a = np.dot(weights3,x[:-1])
                
            
            if x[-1] == "class-1":
                y = 1
            else:
                y = -1


            if y*a <= 0:
                weights3[0] += y
                weights3[1] += y*x[1]
                weights3[2] += y*x[2]
                weights3[3] += y*x[3]
                weights3[4] += y*x[4]
            else:
                correct_class_count3 += 1


            objects3,features = classifier3.shape

            accuracy3 = correct_class_count3/objects3




        # Setting binary classifier for 1vsall approach as required for question 4
        # This classifier will run three times, to allow each of the classes to be the positive class at one point
        # This in turn will provide us 3 weight sets as a result


        #creating multi-class component for question 4....

        for x in trainData:

            x = list(x)
            x.insert(0,1)

            a = np.dot(weights1vAll,x[:-1])

            if x[-1] == "class-1":
                y = 1
            else:
                y = -1
            
            if y*a <= 0:
                weights1vAll[0] += y
                weights1vAll[1] += y*x[1]
                weights1vAll[2] += y*x[2]
                weights1vAll[3] += y*x[3]
                weights1vAll[4] += y*x[4]
            else:
                correct_class_count1vall += 1



            a2 = np.dot(weights2vAll,x[:-1])

            if x[-1] == "class-2":
                y = 1
            else:
                y = -1

            if y*a2 <= 0:
                weights2vAll[0] += y
                weights2vAll[1] += y*x[1]
                weights2vAll[2] += y*x[2]
                weights2vAll[3] += y*x[3]
                weights2vAll[4] += y*x[4]
            else:
                correct_class_count2vall += 1



            a3 = np.dot(weights3vAll,x[:-1])

            if x[-1] == "class-3":
                y = 1
            else:
                y = -1

            if y*a3 <= 0:
                weights3vAll[0] += y
                weights3vAll[1] += y*x[1]
                weights3vAll[2] += y*x[2]
                weights3vAll[3] += y*x[3]
                weights3vAll[4] += y*x[4]
            else:
                correct_class_count3vall += 1




        # Now we can use our multi-class weights to make class predictions
        for x in trainData:

            if x[-1] == "class-1":
                actual_Class = 1
            elif x[-1] == "class-2":
                actual_Class = 2
            elif x[-1] == "class-3":
                actual_Class = 3

            x = list(x)
            x.insert(0,1)

            a = np.dot(weights1vAll,x[:-1]),np.dot(weights2vAll,x[:-1]),np.dot(weights3vAll,x[:-1])

            if np.argmax(a)+1 == actual_Class:
                correct_prediction += 1
        
                
            
            objectsTotal,featuresTotal = trainData.shape

            multi_Class_accuracy = str(correct_prediction/objectsTotal * 100) + "%"
            
    
    
    print("1 vs 2: " + str(weights1) + " accuracy : " + str(accuracy1*100) + "% " + "at iter " + str(iterCount))
    print()
    print("2 vs 3 " + str(weights2)  + " accuracy : " + str(accuracy2*100) + "% " + "at iter " + str(iterCount))
    print()
    print("3 vs 1: " + str(weights3)+ " accuracy : " + str(accuracy3*100) + "% " + "at iter " + str(iterCount))
    print()
    print("Multi-class accuracy after training for: " + str(iterCount) + " iterations is: " + str(multi_Class_accuracy))    


    return weights1,weights2,weights3,weights1vAll,weights2vAll,weights3vAll,multi_Class_accuracy








def perceptronTest(weights1v2,weights2v3,weights1v3,weights1vall,weights2vall,weights3vall,testData):


    # Whilst I could of returned my classifiers from the train perceptron, i did not want to have to pass so much into my test algorithm function
    # As such I decided to simply recreate them here.

    classifier1 = np.delete(testData,np.where(testData == "class-3")[0],axis=0)
    classifier2 = np.delete(testData,np.where(testData == "class-1")[0],axis=0)
    classifier3 = np.delete(testData,np.where(testData == "class-2")[0],axis=0)
    classifierMulti = testData


    #setting correct classification counters for Q3 binary classifiers
    correct_Classification1 = 0
    correct_Classification2 = 0
    correct_Classification3 = 0


    # Setting prediction counters to gauge accuracy for multi-class
    correct_prediction = 0
        



    for x in classifier1:

        if x[-1] == "class-1":
            actual_Object_class = 1
        elif x[-1] == "class-2":
            actual_Object_class = -1

        x = list(x)
        x.insert(0,1)
        a = np.dot(weights1v2,x[:-1])

        if np.sign(a) == actual_Object_class:
            correct_Classification1 += 1

        objects,features = classifier1.shape

        accuracy1 = str(correct_Classification1/objects * 100) + "%"
        
        
    

    for x in classifier2:

        if x[-1] == "class-2":
            actual_Object_class = 1
        elif x[-1] == "class-3":
            actual_Object_class = -1

        x = list(x)
        x.insert(0,1)
        a = np.dot(weights2v3,x[:-1])

        if np.sign(a) == actual_Object_class:
            correct_Classification2 += 1

        objects,features = classifier2.shape

        accuracy2 = str(correct_Classification2/objects * 100) + "%"

    


    for x in classifier3:

        if x[-1] == "class-1":
            actual_Object_class = 1
        elif x[-1] == "class-3":
            actual_Object_class = -1

        x = list(x)
        x.insert(0,1)
        a = np.dot(weights1v3,x[:-1])


        if np.sign(a) == actual_Object_class:
            correct_Classification3 += 1

        objects,features = classifier3.shape

        accuracy3 = str(correct_Classification3/objects * 100) + "%"
    

        
        
    
    for x in testData:

        if x[-1] == "class-1":
            actual_Class = 1
        elif x[-1] == "class-2":
            actual_Class = 2
        elif x[-1] == "class-3":
            actual_Class = 3

        x = list(x)
        x.insert(0,1)

        a = np.dot(weights1vall,x[:-1]),np.dot(weights2vall,x[:-1]),np.dot(weights3vall,x[:-1])

        if np.argmax(a)+1 == actual_Class:
            correct_prediction += 1
            
        objectsTotal,featuresTotal = testData.shape

        multi_Class_accuracy = str(correct_prediction/objectsTotal * 100) + "%"



    print("The test accuracy for binary 1v2: " + str(accuracy1))
    print("The test accuracy for binary 2v3: " + str(accuracy2))
    print("The test accuracy for binary 1v3: " + str(accuracy3))
    print("The test accuracy for Multi-class classification: " + str(multi_Class_accuracy))

    return accuracy1,accuracy2,accuracy3,multi_Class_accuracy  
    


#The functions are then called, and the perceptron's are activated

perceptronTrain(data_Array,20)
perceptronTest(weights1,weights2,weights3,weights1vAll,weights2vAll,weights3vAll,test_Array)

