# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:48:51 2020

@author: rathi
"""
from sympy import Interval,Union
import numpy as np
#from .TrajectoryGroup import TrajectoryGroup
import copy

class Node:
    def __init__(self,bench,subgraph,cost,conflicts,evaluate,children,parent,root,trajectories,subgraphCost):
       self.cost = cost
       self.bench = bench
       self.subgraph = subgraph
       self.conflicts = conflicts
       self.evaluate = evaluate
       self.children = children
       self.parent = parent
       self.root = root
       self.trajectories = trajectories
       self.internalConflicts = False
       self.externalConflicts = True
       self.successChild = None
       self.stops = 0
       self.subgraphCost = subgraphCost
       self.delays = np.zeros(np.size(subgraph))
       if evaluate==False:
          self.evaluateNode()
          
    def evaluateNode(self):
       linkRadius = self.bench.cobras.linkRadius
       replan = True
       count = 0

          
       while(replan):
         self.planTrajectories()
         replan = False
         count = count+1
         for coindex,conflict in enumerate(self.conflicts):          
           distance = self.bench.distancesBetweenLineSegments(self.trajectories.fiberPositions[conflict[0],:], self.trajectories.elbowPositions[conflict[0],:],self.trajectories.fiberPositions[conflict[1],:], self.trajectories.elbowPositions[conflict[1],:])
           self.pairCollision = distance<(linkRadius[conflict[0], np.newaxis] + linkRadius[conflict[1], np.newaxis])
           #print(conflict,any(self.pairCollision))
           if any(self.pairCollision):
              collisionTimes = np.where(self.pairCollision)
              sizeInterval = np.max(collisionTimes)-np.min(collisionTimes)+1
              #sizeInterval = 5
              if self.conflicts[coindex][3]>1:
                self.conflicts[coindex][3] = max(self.conflicts[coindex][3]-sizeInterval,1)
                replan = True
              elif self.conflicts[coindex][4]<self.nSteps-1:
                self.conflicts[coindex][4] = min(self.conflicts[coindex][4]+sizeInterval,self.nSteps-1)
                replan = True
              else:
                self.internalConflicts = True  
              #self.conflicts[coindex][4] = max(np.max(collisionTimes),conflict[4])
              #print(self.conflicts[coindex])
              #print(collisionTimes)
           #if 80 in self.subgraph:
           #  print("Found") 
           #  print(self.subgraph)
           #  print(self.conflicts)
           #  print(self.internalConflicts)     
       
       self.evaluate = True
           
    def planTrajectories(self):
       self.trajectories = copy.deepcopy(self.parent.trajectories) 
       self.nSteps = self.trajectories.tht.shape[1]
       self.cost = self.parent.cost
       self.stops = self.parent.stops
       self.internalConflicts = False
       cobraCenters = self.bench.cobras.centers
       L1 = self.bench.cobras.L1
       L2 = self.bench.cobras.L2
       for nodeindex,node in enumerate(self.subgraph):
         #intervals = []
         times = []
         thtnew = np.array(self.trajectories.tht[node,:])
         phinew = np.array(self.trajectories.phi[node,:])
         elbowPositionsnew = np.array(self.trajectories.elbowPositions[node,:])
         fiberPositionsnew = np.array(self.trajectories.fiberPositions[node,:])
         for conflict in self.conflicts:
           if conflict[2]==node:  
             times.append((conflict[3],'s'))
             if conflict[4]<self.nSteps:
               times.append((conflict[4],'e'))
         times.append((self.nSteps,'s'))  
 
         times.sort(key=lambda x:x[0])
         
         stop = False
         delays = [0]
         intervals = []
         for time in times:
             if time[1]=='s' and stop==False:
                intervals.append(time[0])
                stop = True
             elif time[1]=='e' and stop==True:
                intervals.append(time[0])
                delays.append(intervals[-1]-intervals[-2])
                stop = False
         waitDuration = int(np.max(delays))
         waitStart = int(times[0][0])
         if node in self.parent.subgraph:
             parentnodeindex = np.argwhere(self.parent.subgraph==node).ravel()
             parentwait = self.parent.delays[parentnodeindex]
         self.delays[nodeindex] = waitDuration+parentwait
    
         if waitStart>1 and waitStart<self.nSteps:
            self.stops = self.stops+1 
         
         if self.delays[nodeindex]+waitStart+self.subgraphCost[nodeindex]>self.nSteps and waitStart<self.nSteps:
            #print("Here:",self.subgraph)
            self.internalConflicts = True 
            self.trajectories.tht[node,:] = thtnew[0]
            self.trajectories.phi[node,:] = phinew[0]
            self.trajectories.elbowPositions[node,:] = elbowPositionsnew[0]
            self.trajectories.fiberPositions[node,:] = fiberPositionsnew[0]
            self.cost = self.cost+self.nSteps
         else:
            self.trajectories.tht[node,0:waitStart-1] = thtnew[0:waitStart-1]
            self.trajectories.tht[node,waitStart:waitStart+waitDuration] = thtnew[waitStart-1]
            self.trajectories.tht[node,waitStart+waitDuration:self.nSteps] = thtnew[waitStart:self.nSteps-waitDuration] 
            
            self.trajectories.phi[node,0:waitStart-1] = phinew[0:waitStart-1]
            self.trajectories.phi[node,waitStart:waitStart+waitDuration] = phinew[waitStart-1]
            self.trajectories.phi[node,waitStart+waitDuration:self.nSteps] = phinew[waitStart:self.nSteps-waitDuration] 
            
            #self.trajectories.elbowPositions[node,:] = cobraCenters[node, np.newaxis] + L1[node, np.newaxis] * np.exp(1j * self.trajectories.tht[node,:])
            #self.trajectories.fiberPositions[node,:] = self.trajectories.elbowPositions[node,:]  + L2[node, np.newaxis] * np.exp(1j * (self.trajectories.tht[node,:] + self.trajectories.phi[node,:]))
        
            self.trajectories.elbowPositions[node,0:waitStart-1] = elbowPositionsnew[0:waitStart-1]
            self.trajectories.elbowPositions[node,waitStart:waitStart+waitDuration] = elbowPositionsnew[waitStart-1]
            self.trajectories.elbowPositions[node,waitStart+waitDuration:self.nSteps] = elbowPositionsnew[waitStart:self.nSteps-waitDuration] 
            
            self.trajectories.fiberPositions[node,0:waitStart-1] = fiberPositionsnew[0:waitStart-1]
            self.trajectories.fiberPositions[node,waitStart:waitStart+waitDuration] = fiberPositionsnew[waitStart-1]
            self.trajectories.fiberPositions[node,waitStart+waitDuration:self.nSteps] = fiberPositionsnew[waitStart:self.nSteps-waitDuration] 
            self.cost = self.cost+waitDuration