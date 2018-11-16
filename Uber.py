#This is the start of my Uber 365 major project, lets hope it goes well!
#11/03/18
#Ayrton Foster 

#Checklist of stuff I can improve in the current working algorithm 
#[]: Use A* or Similar Algorith instead of Dijkstras Algorithm
#[]: Save previous results so that they don't need to be recalculated -> maybe in a 2d matrix
    #->[]: make a 2d list of lists as golbal var, in each iteration of the algo, check whether than distance has been calculated yet
    #->[]: Initially check just path comeplted, but later every path that was calculated in djikstras.
#[x]: More efficient assignment of car to passenger if one car is busy and other is not
#[]: Ability for Uber cars to see into future!?!? (see next stop)

#Start uber!
from collections import namedtuple
import csv
import random
from random import randint 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import math
from tqdm import tqdm

#########################################GOLBAL VARIABLES############################

#These are the global variables used in use Dijkstras  
target_node = 0
node_distance = 0

known_paths = [[0 for x in range(50)] for y in range(50)]

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
 
def shortestPath(start_loc, end_loc):
    global known_paths

    if (known_paths[start_loc][end_loc] != 0):
        return known_paths[start_loc][end_loc]
    else:
        dist = useDikstras(start_loc, end_loc)
        known_paths[start_loc][end_loc] = dist
        known_paths[end_loc][start_loc] = dist
        return dist
#This function will compute the A* Algorithm / be the main function from which A* Operates
#It will return the time it took to reach node A from node B 
#def AStarAlgo():
#    ho = 5

class Graph():
 
    def __init__(self, vertices):
        self.adjList = {}   # To store graph: u -> (v,w)
        #self.num_nodes = vertices    # Number of nodes in graph


        self.V = vertices
        self.graph = [[0 for column in range(vertices)] 
                      for row in range(vertices)]


        self.dist = [0] * self.V
        self.par = [-1] * self.V  # To store the path                      
 
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

        #if (node == 10):
        #    return dist[node]
        self.printSolution(dist, src)       

def mainAlgo(start_location, end_location, pickup_time):

    #Initiate main varibles used inthe function 
    car1_location = 0
    car2_location = 0
    car1_time = 0
    car2_time = 0
    tot_wait_time = 0
    pickups_completed = len(pickup_time)
    i = 0

    #OF IMPORTANCE: I have to do startlocation[i] -1 because my node indexes start at 0 - 49 not 1 - 50 

    #pbar = tqdm(total = 300)
    #or when i = pickups_completed
    for i in tqdm(range(300)):
    #while (i < pickups_completed):
        #car1_dist = useDikstras(car1_location, start_location[i]-1)                             #Find the distance from car 1 to the pickup location
        #car2_dist = useDikstras(car2_location, start_location[i]-1)                             #Find the distance from car 2 to the pickup location
        #Section for case where the next pickup request time is greater than both cars current time
        if(pickup_time[i] >  car1_time and pickup_time[i] > car2_time):
            #car1_dist = useDikstras(car1_location, start_location[i]-1)                             #Find the distance from car 1 to the pickup location
            #car2_dist = useDikstras(car2_location, start_location[i]-1)                             #Find the distance from car 2 to the pickup location
            
            car1_dist = shortestPath(car1_location, start_location[i]-1)                             #Find the distance from car 1 to the pickup location
            car2_dist = shortestPath(car2_location, start_location[i]-1)


            #If Car1 is closer to the pickup location than car 2
            if(car1_dist <= car2_dist):
                #car1_location = start_location[i]                                  #update the cars current location 
                #car1_time += (pickup_time[i] - car1_time)                           #Set the fact that the car had to wait until the pickup was requested to move 
                car1_time = pickup_time[i]
                car1_time += car1_dist                                              #Update the cars current time given travel to pickup location 
                tot_wait_time += car1_dist                                          #Update the tot time passengers are waiting for pickup  
                #car1_dist = useDikstras(start_location[i]-1, end_location[i]-1)     #Use Dijkstras to find distance from pickup location to drop off location
                car1_dist = shortestPath(start_location[i]-1, end_location[i]-1)     #Use Dijkstras to find distance from pickup location to drop off location
                car1_time += car1_dist                                              #Update the cars current time given travel to drop off location 
                car1_location = end_location[i]-1                                   #update the cars current location gieb the drop off 
            else:
                #car2_location = start_location[i]                                  #update the cars current location 
                #car2_time += (pickup_time[i] - car1_time)                           #Set the fact that the car had to wait until the pickup was requested to move 
                car2_time = pickup_time[i]                
                car2_time += car2_dist                                              #Update the cars current time given travel to pickup location 
                tot_wait_time += car2_dist                                          #Update the tot time passengers are waiting for pickup  
                #car2_dist = useDikstras(start_location[i]-1, end_location[i]-1)     #Use Dijkstras to find distance from pickup location to drop off location
                car2_dist = shortestPath(start_location[i]-1, end_location[i]-1)     #Use Dijkstras to find distance from pickup location to drop off location
                car2_time += car2_dist                                              #Update the cars current time given travel to drop off location 
                car2_location = end_location[i]-1                                   #update the cars current location gieb the drop off 
                

            i+=1                         #update the current index of next job that needs to be taken 

        #section where only one cars current time is less than next pickup request time
        #When car 1's current time is before next pickup, and car 2 is after
        elif(pickup_time[i] >= car1_time and pickup_time[i] < car2_time):
            car1_dist = shortestPath(car1_location, start_location[i]-1)             #Calculate time from car1 to pickup location  
            #car1_time += ((pickup_time[i]) - car1_time)                             #Take into account time car1 must wait for pickup request  
            car1_time = pickup_time[i]
            car1_time += car1_dist                                                #Add time required for car1 t reach pickup to cars time  
            tot_wait_time += car1_dist                                            #Add time it took for car to reach passenger to tot_wait_time
            car1_dist = shortestPath(start_location[i]-1, end_location[i]-1)           #Calculate time it takes for car to complate drop off  
            car1_time += car1_dist                                                #Add time take to reach destination to car1 time  
            car1_location = end_location[i] -1                                       #Set car1's location to the drop off location 
            i+=1 


        #When car 2's current time is before next pickup, and car 1 is after        
        elif(pickup_time[i] < car1_time and pickup_time[i] >= car2_time):
            car2_dist = shortestPath(car2_location, start_location[i]-1)             #Calculate time from car2 to pickup location  
            #car2_time += (pickup_time[i] - car2_time)                             #Take into account time car2 must wait for pickup request  
            car2_time = pickup_time[i]                            
            car2_time += car2_dist                                                #Add time required for car2 t reach pickup to cars time  
            tot_wait_time += car2_dist                                            #Add time it took for car to reach passenger to tot_wait_time
            car2_dist = shortestPath(start_location[i]-1, end_location[i]-1)           #Calculate time it takes for car to complate drop off  
            car2_time += car2_dist                                                #Add time take to reach destination to car2 time  
            car2_location = end_location[i]-1                                       #Set car2's location to the drop off location 
            i+=1  

        #section where both cars current time is greater than the next pickup time
            #Choose the car with the lower current time as pickup car
        else:
            car1_dist = shortestPath(car1_location, start_location[i]-1)                             #Find the distance from car 1 to the pickup location
            car2_dist = shortestPath(car2_location, start_location[i]-1)             

            #if(car1_time <= car2_time):
            #if(car1_dist <= car2_dist):
            if((car1_time + car1_dist) <= (car2_time + car2_dist)):
                #dikstras car 1
                car1_dist = shortestPath(car1_location, start_location[i]-1)                   #Find the distance from car 1 to the pickup location
                #car1_time += (pickup_time[i] - car1_time)                                  #Take into account time car1 must wait for pickup request 
                tot_wait_time += (car1_dist + (car1_time-pickup_time[i]))                     #Add time it took for car to reach passenger to tot_wait_time
                car1_time += car1_dist                                                      #Add time required for car1 t reach pickup to cars time  
                car1_dist = shortestPath(start_location[i]-1, end_location[i]-1)                 #Calculate time it takes for car to complate drop off  
                car1_time += car1_dist                                                      #Add time take to reach destination to car1 time  
                car1_location = end_location[i]-1                                             #Set car1's location to the drop off location 

            else:
                #dikstras car 2
                car2_dist = shortestPath(car2_location, start_location[i]-1)             
                #car2_time += (pickup_time[i] - car2_time)                                  #Take into account time car1 must wait for pickup request 
                tot_wait_time += (car1_dist + (car1_time-pickup_time[i]))                     #Add time it took for car to reach passenger to tot_wait_time                
                car2_time += car2_dist                                                      #Add time required for car1 t reach pickup to cars time  
                car2_dist = shortestPath(start_location[i]-1, end_location[i]-1)                 #Calculate time it takes for car to complate drop off  
                car2_time += car2_dist                                                      #Add time take to reach destination to car1 time  
                car2_location = end_location[i]-1                                             #Set car1's location to the drop off location 

            i+=1 
        
        #print("car ride ",i," completed, tot_wait_time ", tot_wait_time)       
        #print("car ride ",i," completed")                                                                               #update the current index of next job that needs to be taken 
                                                                        #update the current index of next job that needs to be taken 

    return tot_wait_time

    

####################################NEW CODE STUFF########################

####################################END NEW CODE STUFF########################



###################################START OF MAIN FUNCTION THAT WILL CALL OTHER FUNTIONS##############################

network = readNetworkFile("network.csv")
#print(network)
#print([0][0])

start_location, end_location, pickup_time = readRequestsFile("requests.csv")
#start_location, end_location, pickup_time = readRequestsFile("supplementpickups.csv")
#print(start_location)
#print(end_location)
#print(pickup_time)

# Driver program
g  = Graph(50)
g.graph = network
'''g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
            [4, 0, 8, 0, 0, 0, 0, 11, 0],
            [0, 8, 0, 7, 0, 4, 0, 0, 2],
            [0, 0, 7, 0, 9, 14, 0, 0, 0],
            [0, 0, 0, 9, 0, 10, 0, 0, 0],
            [0, 0, 4, 14, 10, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 1, 6],
            [8, 11, 0, 0, 0, 0, 1, 0, 7],
            [0, 0, 2, 0, 0, 0, 6, 7, 0]
            ]'''
    
#g.dijkstra(0)
#print("distance from 0 to ",target_node," is ",node_distance)
#g.show_path(0,10)

#blah = useDikstras(8,46)
#print("distance from 8 to ",target_node," is ", blah)

#distance = shortestPath(8, 46)
#print(distance)
#distance = shortestPath(8, 46)
#print(distance)

time_waiting = mainAlgo(start_location, end_location, pickup_time)
print(time_waiting)

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




############################################RANDOM CODE###############################

'''
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far
'''

#######################################SEPERATE BETWEEN DIFFERENT OLD CODE################

'''
class PriorityQueue:
    # Based on Min Heap
    def __init__(self):
        self.cur_size = 0
        self.array = []
        self.pos = {}   # To store the pos of node in array

    def isEmpty(self):
        return self.cur_size == 0

    def min_heapify(self, idx):
        lc = self.left(idx)
        rc = self.right(idx)
        if lc < self.cur_size and self.array(lc)[0] < self.array(idx)[0]:
            smallest = lc
        else:
            smallest = idx
        if rc < self.cur_size and self.array(rc)[0] < self.array(smallest)[0]:
            smallest = rc
        if smallest != idx:
            self.swap(idx, smallest)
            self.min_heapify(smallest)

    def insert(self, tup):
        # Inserts a node into the Priority Queue
        self.pos[tup[1]] = self.cur_size
        self.cur_size += 1
        self.array.append((sys.maxsize, tup[1]))
        self.decrease_key((sys.maxsize, tup[1]), tup[0])

    def extract_min(self):
        # Removes and returns the min element at top of priority queue
        min_node = self.array[0][1]
        self.array[0] = self.array[self.cur_size - 1]
        self.cur_size -= 1
        self.min_heapify(1)
        del self.pos[min_node]
        return min_node

    def left(self, i):
        # returns the index of left child
        return 2 * i + 1

    def right(self, i):
        # returns the index of right child
        return 2 * i + 2

    def par(self, i):
        # returns the index of parent
        return math.floor(i / 2)

    def swap(self, i, j):
        # swaps array elements at indices i and j
        # update the pos{}
        self.pos[self.array[i][1]] = j
        self.pos[self.array[j][1]] = i
        temp = self.array[i]
        self.array[i] = self.array[j]
        self.array[j] = temp

    def decrease_key(self, tup, new_d):
        idx = self.pos[tup[1]]
        # assuming the new_d is atmost old_d
        self.array[idx] = (new_d, tup[1])
        while idx > 0 and self.array[self.par(idx)][0] > self.array[idx][0]:
            self.swap(idx, self.par(idx))
            idx = self.par(idx)


class Graph:
    def __init__(self, num):
        self.adjList = {}   # To store graph: u -> (v,w)
        self.num_nodes = num    # Number of nodes in graph
        # To store the distance from source vertex
        self.dist = [0] * self.num_nodes
        self.par = [-1] * self.num_nodes  # To store the path

    def add_edge(self, u, v, w):
        #  Edge going from node u to v and v to u with weight w
        # u (w)-> v, v (w) -> u
        # Check if u already in graph
        if u in self.adjList.keys():
            self.adjList[u].append((v, w))
        else:
            self.adjList[u] = [(v, w)]

        # Assuming undirected graph
        if v in self.adjList.keys():
            self.adjList[v].append((u, w))
        else:
            self.adjList[v] = [(u, w)]

    def show_graph(self):
        # u -> v(w)
        for u in self.adjList:
            print(u, '->', ' -> '.join(str("{}({})".format(v, w))
                                       for v, w in self.adjList[u]))

    def dijkstra(self, src):
        # Flush old junk values in par[]
        self.par = [-1] * self.num_nodes
        # src is the source node
        self.dist[src] = 0
        Q = PriorityQueue()
        Q.insert((0, src))  # (dist from src, node)
        for u in self.adjList.keys():
            if u != src:
                self.dist[u] = sys.maxsize  # Infinity
                self.par[u] = -1

        while not Q.isEmpty():
            u = Q.extract_min()  # Returns node with the min dist from source
            # Update the distance of all the neighbours of u and
            # if their prev dist was INFINITY then push them in Q
            for v, w in self.adjList[u]:
                new_dist = self.dist[u] + w
                if self.dist[v] > new_dist:
                    if self.dist[v] == sys.maxsize:
                        Q.insert((new_dist, v))
                    else:
                        Q.decrease_key((self.dist[v], v), new_dist)
                    self.dist[v] = new_dist
                    self.par[v] = u

        # Show the shortest distances from src
        self.show_distances(src)

    def show_distances(self, src):
        print("Distance from node: {}".format(src))
        for u in range(self.num_nodes):
            print('Node {} has distance: {}'.format(u, self.dist[u]))

    def show_path(self, src, dest):
        # To show the shortest path from src to dest
        # WARNING: Use it *after* calling dijkstra
        path = []
        cost = 0
        temp = dest
        # Backtracking from dest to src
        while self.par[temp] != -1:
            path.append(temp)
            if temp != src:
                for v, w in self.adjList[temp]:
                    if v == self.par[temp]:
                        cost += w
                        break
            temp = self.par[temp]
        path.append(src)
        path.reverse()

        print('----Path to reach {} from {}----'.format(dest, src))
        for u in path:
            print('{}'.format(u), end=' ')
            if u != dest:
                print('-> ', end='')

        print('\nTotal cost of path: ', cost)
'''