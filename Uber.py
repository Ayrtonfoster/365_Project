#This is the start of my Uber 365 major project, lets hope it goes well!
#11/03/18
#Ayrton Foster 

#Start uber!
from collections import namedtuple
import csv
import random
from random import randint 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

#will now try to outline how the algorithm will work


#This function will take the data from the teh requests.csv and returns all three columns as lists
def readRequestsFile(filename):
    df = pd.read_csv(filename, names=['start', 'end', 'timestamp'])
    start_location = df.start
    end_location = df.end
    pickup_time = df.timestamp
    return start_location, end_location, pickup_time


#This function will take the data from data from network.csv 
def readNetworkFile(filename): 
    df = pd.read_csv(filename, header=None)
    #m = df.unstack()
    #m = m.fillna(0)
    #print (m)
    return df

#This function will compute the A* Algorithm / be the main function from which A* Operates
#It will return the time it took to reach node A from node B 
def AStarAlgo():
    ho = 5


###################################START OF MAIN FUNCTION THAT WILL CALL OTHER FUNTIONS##############################

network = readNetworkFile("network.csv")
print(network)
#print([0][0])

start_location, end_location, pickup_time = readRequestsFile("requests.csv")
#print(start_time)
#print(end_time)
#print(pickup_time)


###################################END OF START OF MAIN FUNCTION THAT WILL CALL OTHER FUNTIONS##############################


#######################OUTLINING ALGORITHM###############

#1) Call a function that that will return the three columns in requests.csv, each as a python list
    #i) Pickup Request Time (ie: time pick up is requested)
    #ii) The Picket Location (ie: where the passenger needs to be picked up)
    #iii) The Dropoff location (ie: where the passenger wants to be dropped off)

#2) Call a function that will return the contents of network.csv into a format that can be used by a pathing algo
    #i) The data will be stored in an uknown format -> likely as a python 2d list 

#3) Initialize the global variables that may be needed for this algorithm
    #i) graph, start_time
    #ii) non global vars car1time, car2time, waitTime, CALocation, CBLocation      

''' Tentative step (not sure if I should do this now) Not everything in step4 is in order  '''
#4) Start the task by having each of the two "cars" choose the first two pickup requests, picking them up, then dropping them off
''' Here a sorting algorithm will be given the STARTING and ENDING node '''
''' It will the pickup time, travel to destination time, and total time '''
    #i) Of note: I will assume both cars start at node 0 

    #ii) In each Algorith, record down how long it took each car to 
        #A) reach the passenger 
        #B) Then seperatly how long it took to deliver them to the drop of location  
        #C) In each ride add the time to pickup (ttp) to the continously current amount
        #D) Record down how long it to complete the whole pickup and drop off task
        #E) Finally record down the location that the car dopped its passengers off at     
''' Initial case over, begin looping through all the different requests '''
''' The pathing loop will be called twice for each request -> Path to reach pickup location, then to passenger destination '''
    #iii) Compare the overall time it took each "car" to complete its run
        #A) The Car with the lower overall time will take the next pickup request with the lowest time 
        #B) Pass the starting and ending node to the pathing algorithm
        #C) Pathing algorithm will return the time it took to rach the starting node to the destination node  
        #D) Add the total time to the cars current concept of time 
        #E) Add the wait time to the master wait time variable + time it took for uber to just respond to request
        #F) Return the resting node of the "car" to its resting_car_location variable 
        #G) Call pathing algorith again, this time for pickup -> dropoff, return same vars and perform same actions   

    #IV) Check the current time of "car1" vs "car2" again choose car to perfrom next pickup based on lower "current time"   
        #A) Repeat the process iii) with the new ride request 
        
    #V) Repeat the algorithm until all pickup requests have been fulfilled 
        #A) return the overall time it took to complete all Runs, as well as total passenger wait time

#5) Print out the results to the screen!
    #i) Easy!                         



#Imortant notes:
    # When a car completes a "route" (ie responds to a pickup requests, pickups passenger, delivers passenger to dropp off location)
    # The total wait time for the passenger is 
        #i) The time it took for the uber to aknowledge the pickup request (uber current time - pickup request time)
        #ii) + the time it took for the uber to reach the passenger

    # The total wait time for passengers will build up if the pathing algorithm is efficient, 
    # Since the more time spent drving means more requests will build up over time.     



##########################################QUESTIONS##################################

#1) What Pathing algorithm should we use? 
#2) How do we represent the data in a form the pathing algorith can understand? 
#3) How can we make the algorithm even more efficient -> decrease wait times (Assuming algo works)
    #i) Dont necessarly choose cars for trips based on availability, maybe on proximity to pickup location?
#4) How will we divide this algorithm up in terms of functions

##########################################END QUESTIONS###############################    