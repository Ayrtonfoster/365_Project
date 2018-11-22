#This is the start of my Uber 365 major project, lets hope it goes well!
#11/03/18
#Ayrton Foster 

#Checklist of stuff I can improve in the current working algorithm 
#[]: Use A* or Similar Algorith instead of Dijkstras Algorithm
#[x]: Finish Check data function
#[x]: Save previous results so that they don't need to be recalculated -> maybe in a 2d matrix
    #->[x]: make a 2d list of lists as golbal var, in each iteration of the algo, check whether than distance has been calculated yet
    #->[x]: Initially check just path comeplted, but later every path that was calculated in djikstras.
#[x]: More efficient assignment of car to passenger if one car is busy and other is not
#[x]: Ability for Uber cars to see into future!?!? (see next stop)

import math
import random
import sys
#Start uber!
from collections import namedtuple
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

#########################################GOLBAL VARIABLES############################

#These are the global variables used in use Dijkstras  
target_node = 0
node_distance = 0
known_paths = [[0 for x in range(50)] for y in range(50)]

heatmap = [0 for x in range(50)]
#########################################GOLBAL VARIABLES############################


#This function will take the data from the teh requests.csv and returns all three columns as lists
def readRequestsFile(filename):
    df = pd.read_csv(filename, names=['timestamp', 'start', 'end'])
    pickup_time = df.timestamp 
    start_location = df.start
    end_location = df.end
    return start_location, end_location, pickup_time


#This function will take the data from data from network.csv 
def readNetworkFile(filename): 
    df = pd.read_csv(filename, header=None)
    return df


#This function jerry rigs the dijkstras algo to return the dist from orig to target node
def useDikstras(start_loc, end_loc):
    #Intialize golbal variables used in function
    global target_node
    global node_distance

    target_node = end_loc    # Set Global value for end location
    g.dijkstra(start_loc)     # call dikstras algorithm with start loc
    dist = node_distance     # save global distance var to local var
    return dist         # return local var
 
# This function checks whether the shortest path desired has been calculated already  
def shortestPath(start_loc, end_loc):
    global known_paths

    #Of the path has already been calculated, return the distance
    if (known_paths[start_loc][end_loc] != 0):
        return known_paths[start_loc][end_loc]
    
    #If the distance hasn't been calculated yet, calculate it then store its distance
    else:
        dist = useDikstras(start_loc, end_loc)
        #known_paths[start_loc][end_loc] = dist
        #known_paths[end_loc][start_loc] = dist
        return dist

#Dijkstras algorithm derived and moified from original algorithm found here
#https://github.com/OpenGenus/cosmos/blob/master/code/graph_algorithms/src/dijkstra_shortest_path/Dijkstra.py
class Graph():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] 
                      for row in range(vertices)]                    
 
    def printSolution(self, dist, src):
        global target_node
        global node_distance
        global known_paths
        #print ("Vertex tDistance from Source")
        for node in range(self.V):
            known_paths[src][node] = dist[node]
            known_paths[node][src] = dist[node]
            #print (node,"t",dist[node])
            #print(self.V)
            #if(node == target_node):
                #node_distance = dist[node]

        node_distance = dist[target_node]        
 
    # A utility function to find the vertex with 
    # minimum distance value, from the set of vertices 
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initilaize minimum distance for next node
        min = sys.maxsize
 
        # Search not nearest vertex not in the 
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index
 
    # Funtion that implements Dijkstra's single source 
    # shortest path algorithm for a graph represented 
    # using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from 
            # the set of vertices not yet processed. 
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the 
            # shotest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices 
            # of the picked vertex only if the current 
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]):
                        dist[v] = dist[u] + self.graph[u][v]

        self.printSolution(dist, src)       

#checks if there is a valid number
def check_if_there_is_data(start_location, end_location, pickup_time, i):

    #If the data within is not a number, skip this data set  
    if(math.isnan(start_location[i]-1) or math.isnan(end_location[i]-1) or math.isnan(pickup_time[i])):    
        i+=1
    return i

# This function performs the update on a cars information after a decision regarding which car should pikcup the passenger has been made
# This function is used for cars whose curren time is before the next pickup time  
def updateInfo1(car_time, car_location, car_dist, tot_wait_time, pickup_time, start_location, end_location, i):
    car_time = pickup_time[i]                                                  #Set the cars time to the current time -> wait until request is made before pickup   
    car_time += car_dist                                                      #Update the cars current time given travel to pickup location 
    tot_wait_time += car_dist                                                  #Update the tot time passengers are waiting for pickup
    car_dist = shortestPath(start_location[i]-1, end_location[i]-1)            #Use Dijkstras to find distance from pickup location to drop off location
    car_time += car_dist                                                      #Update the cars current time given travel to drop off location 
    car_location = end_location[i]-1                                           #Set the cars current location to the drop off location
    i+=1                                                                        #Increment counter to the next pickup request    
    return car_time, tot_wait_time, car_dist, car_location, i            

# This function performs the update on a cars information after a decision regarding which car should pikcup the passenger has been made
# This function is used for cars whose curren time is beyond the next pickup time  
def updateInfo2(car_time, car_location, car_dist, tot_wait_time, pickup_time, start_location, end_location, i):
    tot_wait_time += (car_dist + (car_time-pickup_time[i]))                   #Add time it took for car to reach passenger to tot_wait_time
    car_time += car_dist                                                      #Add time required for car1 t reach pickup to cars time  
    car_dist = shortestPath(start_location[i]-1, end_location[i]-1)            #Calculate time it takes for car to complate drop off  
    car_time += car_dist                                                      #Add time take to reach destination to car1 time  
    car_location = end_location[i]-1                                           #Set car1's location to the drop off location 
    i+=1                                                                        #Increment counter to nect pickup requst 
    return car_time, tot_wait_time, car_dist, car_location, i 


def mainAlgo(start_location, end_location, pickup_time):

    #Initiate main varibles used in the function, variable names are self explainatory  
    #global heatmap    
    car1_location = 0
    car2_location = 0
    car1_time = 0
    car2_time = 0
    tot_wait_time = 0
    pickups_completed = len(pickup_time)
    i = 0

    # check of the data given to the algorithm 
    if(len(start_location) != len(end_location) != len(pickup_time)):
        print("invalid data set")
        return i

    #for i in tqdm(range(300)):                                                                  #To see a visual progress bar represent the progress of the algo
    while (i < pickups_completed):

        heatmap[start_location[i]-1] += 1
        
        #check that the next pickup data is valid 
        i = check_if_there_is_data(start_location, end_location, pickup_time, i)

        car1_dist = shortestPath(car1_location, start_location[i]-1)                             #Find the distance from car 1 to the pickup location
        car2_dist = shortestPath(car2_location, start_location[i]-1)                             #Find the distance from car 2 to the pickup location
        
        #If both car 1 and car 2's time is less then the next pickup time         
        if(car1_time <= pickup_time[i] and car2_time <= pickup_time[i]):
            if(car1_dist <= car2_dist):
                car1_time, tot_wait_time, car1_dist, car1_location, i = updateInfo1(
                    car1_time, car1_location, car1_dist, tot_wait_time, pickup_time, start_location, end_location, i)
            else:
                car2_time, tot_wait_time, car2_dist, car2_location, i = updateInfo1(
                    car2_time, car2_location, car2_dist, tot_wait_time, pickup_time, start_location, end_location, i)

        #IF both car 1 and car 2's time is greater then the next pickup time         
        elif(pickup_time[i] <= car1_time and pickup_time[i] <= car2_time):
            if((car1_time + car1_dist) <= (car2_time + car2_dist)):
                car1_time, tot_wait_time, car1_dist, car1_location, i = updateInfo2(
                    car1_time, car1_location, car1_dist, tot_wait_time, pickup_time, start_location, end_location, i)
            else:
                car2_time, tot_wait_time, car2_dist, car2_location, i = updateInfo2(
                    car2_time, car2_location, car2_dist, tot_wait_time, pickup_time, start_location, end_location, i)
        
        #When car 1's current time is before next pickup, and car 2 is after
        elif(pickup_time[i] >= car1_time and pickup_time[i] < car2_time):
            if (car1_dist + pickup_time[i] <= car2_dist + (car2_time - pickup_time[i]) + pickup_time[i]):
                car1_time, tot_wait_time, car1_dist, car1_location, i = updateInfo1(
                    car1_time, car1_location, car1_dist, tot_wait_time, pickup_time, start_location, end_location, i)

            elif(car1_dist + pickup_time[i] > car2_dist + (car2_time - pickup_time[i]) + pickup_time[i]):
                car2_time, tot_wait_time, car2_dist, car2_location, i = updateInfo2(
                    car2_time, car2_location, car2_dist, tot_wait_time, pickup_time, start_location, end_location, i)

        #When car 2's current time is before next pickup, and car 1 is after
        elif(pickup_time[i] < car1_time and pickup_time[i] >= car2_time):
            if (car2_dist + pickup_time[i] <= car1_dist + (car1_time - pickup_time[i]) + pickup_time[i]):
                car2_time, tot_wait_time, car2_dist, car2_location, i = updateInfo1(
                    car2_time, car2_location, car2_dist, tot_wait_time, pickup_time, start_location, end_location, i)

            elif(car2_dist + pickup_time[i] > car1_dist + (car1_time - pickup_time[i]) + pickup_time[i]):
                car2_time, tot_wait_time, car2_dist, car2_location, i = updateInfo2(
                    car2_time, car2_location, car2_dist, tot_wait_time, pickup_time, start_location, end_location, i)

        else:
            print("different case")
          
       
        #print out the car ride # that teh algorithm is currently on 
        print("car ride ",i," completed")

    #print("hottest node val is: ",max(heatmap), "at index: ", index(max(heatmap)))     

    #print(maxIndex)
    #print(heatmap)
    #return the total wait time
    return tot_wait_time


###################################START OF MAIN FUNCTION THAT WILL CALL OTHER FUNTIONS##############################

#Read the start location, end location, and oickup time from file
#Don't have method to select which file to read since it would be a waste of resources

#start_location, end_location, pickup_time = readRequestsFile("requests.csv")

#start_location, end_location, pickup_time = readRequestsFile("supplementpickups.csv")
start_location, end_location, pickup_time = readRequestsFile("requests2.csv")

#Setup graph itself
#network = readNetworkFile("network.csv")                            #read graph weighted adj matrix from file
 
#network = readNetworkFile("network2.csv")
network = readNetworkFile("newnetwork.csv")
network_size = len(network)

g  = Graph(network_size)                                                      #setup graph size to the desired size
g.graph = network                                                   #Assign the graph data received from xml file
#This calls the mian function that will compute the waiting time of the passengers 
time_waiting = mainAlgo(start_location, end_location, pickup_time)
print("total waiting time is: ",time_waiting)

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

