# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:48:51 2020

@author: rathi
"""
from sympy import Interval,Union
import numpy as np
#from .TrajectoryGroup import TrajectoryGroup
import copy

# Find earliest possible stop point with no collision
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
       self.nSteps = self.parent.trajectories.tht.shape[1]
       self.trajectories = copy.deepcopy(self.root.trajectories)

       # Append the conflicts in the parent node with the current node
       if np.size(self.parent.conflicts)>1:
         self.conflicts = np.concatenate((self.conflicts,self.parent.conflicts),axis=0)
      
       allplannodes = np.unique(self.conflicts[:,2])
       
       for node in allplannodes:
           waitStartarr = np.arange(1,self.nSteps-9,25,dtype=np.int)
           nodeindex = np.argwhere(self.subgraph==node).ravel()
           waitDurationarr = np.arange(5,self.nSteps-self.subgraphCost[nodeindex],20,dtype=np.int)
           success = False
           for waitStart in waitStartarr:
             for waitDuration in waitDurationarr:
               # Stop both theta and phi motors  
               outtrajectories = self.planTrajectories(int(node),waitStart,waitDuration,self.trajectories)
               csuccess = True
               # Check whether all conflicts involving the cobra are resolved 
               for conflict in self.conflicts:
                 if conflict[0]==node or conflict[1]==node:  
                   distance = self.bench.distancesBetweenLineSegments(outtrajectories.fiberPositions[conflict[0],:], outtrajectories.elbowPositions[conflict[0],:],outtrajectories.fiberPositions[conflict[1],:], outtrajectories.elbowPositions[conflict[1],:])
                   self.pairCollision = distance<(linkRadius[conflict[0], np.newaxis] + linkRadius[conflict[1], np.newaxis])
                   if any(self.pairCollision):
                     csuccess = False
               if csuccess==True:
                 if waitStart>1:
                     self.stops = self.stops+1
                 success = True
                 self.trajectories = copy.deepcopy(outtrajectories)
                 break;
                 
               # Stop only phi motor  
               outtrajectories2 = self.planTrajectoriesphionly(int(node),waitStart,waitDuration,self.trajectories)
               csuccess = True
               for conflict in self.conflicts:
                 if conflict[0]==node or conflict[1]==node:  
                   distance = self.bench.distancesBetweenLineSegments(outtrajectories2.fiberPositions[conflict[0],:], outtrajectories2.elbowPositions[conflict[0],:],outtrajectories2.fiberPositions[conflict[1],:], outtrajectories2.elbowPositions[conflict[1],:])
                   self.pairCollision = distance<(linkRadius[conflict[0], np.newaxis] + linkRadius[conflict[1], np.newaxis])
                   if any(self.pairCollision):
                     csuccess = False
               if csuccess==True:
                 if waitStart>1:
                     self.stops = self.stops+1  
                 success = True
                 self.trajectories = copy.deepcopy(outtrajectories2)
                 break;  
             if success==True:
                break;
               #self.conflicts[coindex][3] = waitStart
               #self.conflicts[coindex][4] = waitStart+waitDuration
           #if success==False:
           #  for waitStart in waitStartarr:
           #    for waitDuration in waitDurationarr:
           #      for waitStarttht in range(1,waitStart,5):
           #          outtrajectories3 = self.planTrajectoriesthetaphi(int(node),waitStart,waitStarttht,waitDuration,self.trajectories)
           #          csuccess = True
           #          for conflict in self.conflicts:
           #            if conflict[0]==node or conflict[1]==node:  
           #              distance = self.bench.distancesBetweenLineSegments(outtrajectories3.fiberPositions[conflict[0],:], outtrajectories3.elbowPositions[conflict[0],:],outtrajectories3.fiberPositions[conflict[1],:], outtrajectories3.elbowPositions[conflict[1],:])
           #              self.pairCollision = distance<(linkRadius[conflict[0], np.newaxis] + linkRadius[conflict[1], np.newaxis])
           #              if any(self.pairCollision):
           #                csuccess = False
           #          if csuccess==True:
           #            if waitStart>1:
           #              self.stops = self.stops+1
           #            success = True
           #            self.trajectories = copy.deepcopy(outtrajectories)
           #            break;
           #      if success==True:
           #         break;
           #    if success==True:
           #         break;
           if success==False:
              self.internalConflicts = True
              #self.trajectories = copy.deepcopy(self.root.trajectories)
              break;
       self.evaluate = True
    
    # Find the trajectory obtained by delaying the shortest path of node for waitDuration at waitStart       
    def planTrajectories(self,node,waitStart,waitDuration,intrajectories):
       trajectories = copy.deepcopy(intrajectories) 
       self.nSteps = self.trajectories.tht.shape[1]
       self.cost = self.parent.cost
       self.stops = self.parent.stops
       self.internalConflicts = False
       
       thtnew = np.array(trajectories.tht[node,:])
       phinew = np.array(trajectories.phi[node,:])
       elbowPositionsnew = np.array(trajectories.elbowPositions[node,:])
       fiberPositionsnew = np.array(trajectories.fiberPositions[node,:])

       trajectories.tht[node,0:waitStart-1] = thtnew[0:waitStart-1]
       trajectories.tht[node,waitStart:waitStart+waitDuration] = thtnew[waitStart-1]
       trajectories.tht[node,waitStart+waitDuration:self.nSteps] = thtnew[waitStart:self.nSteps-waitDuration] 
            
       trajectories.phi[node,0:waitStart-1] = phinew[0:waitStart-1]
       trajectories.phi[node,waitStart:waitStart+waitDuration] = phinew[waitStart-1]
       trajectories.phi[node,waitStart+waitDuration:self.nSteps] = phinew[waitStart:self.nSteps-waitDuration] 
            
            #self.trajectories.elbowPositions[node,:] = cobraCenters[node, np.newaxis] + L1[node, np.newaxis] * np.exp(1j * self.trajectories.tht[node,:])
            #self.trajectories.fiberPositions[node,:] = self.trajectories.elbowPositions[node,:]  + L2[node, np.newaxis] * np.exp(1j * (self.trajectories.tht[node,:] + self.trajectories.phi[node,:]))
        
       trajectories.elbowPositions[node,0:waitStart-1] = elbowPositionsnew[0:waitStart-1]
       trajectories.elbowPositions[node,waitStart:waitStart+waitDuration] = elbowPositionsnew[waitStart-1]
       trajectories.elbowPositions[node,waitStart+waitDuration:self.nSteps] = elbowPositionsnew[waitStart:self.nSteps-waitDuration] 
            
       trajectories.fiberPositions[node,0:waitStart-1] = fiberPositionsnew[0:waitStart-1]
       trajectories.fiberPositions[node,waitStart:waitStart+waitDuration] = fiberPositionsnew[waitStart-1]
       trajectories.fiberPositions[node,waitStart+waitDuration:self.nSteps] = fiberPositionsnew[waitStart:self.nSteps-waitDuration] 
       return trajectories
    
    # Find the trajectory obtained by delaying only the phiMotor for waitDuration at waitStart       
    def planTrajectoriesphionly(self,node,waitStart,waitDuration,intrajectories):
       trajectories = copy.deepcopy(intrajectories) 
       self.nSteps = self.trajectories.tht.shape[1]
       self.cost = self.parent.cost
       self.stops = self.parent.stops
       self.internalConflicts = False
       cobraCenters = np.array(self.bench.cobras.centers)
       L1 = np.array(self.bench.cobras.L1)
       L2 = np.array(self.bench.cobras.L2)
       
       #thtnew = np.array(trajectories.tht[node,:])
       phinew = np.array(trajectories.phi[node,:])
       #elbowPositionsnew = np.array(trajectories.elbowPositions[node,:])
       #fiberPositionsnew = np.array(trajectories.fiberPositions[node,:])
         
       trajectories.phi[node,0:waitStart-1] = phinew[0:waitStart-1]
       trajectories.phi[node,waitStart:waitStart+waitDuration] = phinew[waitStart-1]
       trajectories.phi[node,waitStart+waitDuration:self.nSteps] = phinew[waitStart:self.nSteps-waitDuration] 
            
       trajectories.elbowPositions[node,:] = cobraCenters[node, np.newaxis] + L1[node, np.newaxis] * np.exp(1j * trajectories.tht[node,:])
       trajectories.fiberPositions[node,:] = trajectories.elbowPositions[node,:]  + L2[node, np.newaxis] * np.exp(1j * (trajectories.tht[node,:] + trajectories.phi[node,:]))
       
       return trajectories
   
    #Unused
    def planTrajectoriesthetaonly(self,node,waitStart,waitDuration,intrajectories):
       trajectories = copy.deepcopy(intrajectories) 
       self.nSteps = self.trajectories.tht.shape[1]
       self.cost = self.parent.cost
       self.stops = self.parent.stops
       self.internalConflicts = False
       cobraCenters = np.array(self.bench.cobras.centers)
       L1 = np.array(self.bench.cobras.L1)
       L2 = np.array(self.bench.cobras.L2)
       
       thtnew = np.array(trajectories.tht[node,:])
       #phinew = np.array(trajectories.phi[node,:])
       #elbowPositionsnew = np.array(trajectories.elbowPositions[node,:])
       #fiberPositionsnew = np.array(trajectories.fiberPositions[node,:])
 
            
       trajectories.tht[node,0:waitStart-1] = thtnew[0:waitStart-1]
       trajectories.tht[node,waitStart:waitStart+waitDuration] = thtnew[waitStart-1]
       trajectories.tht[node,waitStart+waitDuration:self.nSteps] = thtnew[waitStart:self.nSteps-waitDuration] 
            
       trajectories.elbowPositions[node,:] = cobraCenters[node, np.newaxis] + L1[node, np.newaxis] * np.exp(1j * trajectories.tht[node,:])
       trajectories.fiberPositions[node,:] = trajectories.elbowPositions[node,:]  + L2[node, np.newaxis] * np.exp(1j * (trajectories.tht[node,:] + trajectories.phi[node,:]))
       
       return trajectories
    
    #Unused
    def planTrajectoriesthetaphi(self,node,waitStart,waitStarttht,waitDuration,intrajectories):
       trajectories = copy.deepcopy(intrajectories) 
       self.nSteps = self.trajectories.tht.shape[1]
       self.cost = self.parent.cost
       self.stops = self.parent.stops
       self.internalConflicts = False
       cobraCenters = np.array(self.bench.cobras.centers)
       L1 = np.array(self.bench.cobras.L1)
       L2 = np.array(self.bench.cobras.L2)
       
       thtnew = np.array(trajectories.tht[node,:])
       phinew = np.array(trajectories.phi[node,:])
       #elbowPositionsnew = np.array(trajectories.elbowPositions[node,:])
       #fiberPositionsnew = np.array(trajectories.fiberPositions[node,:])
       
       trajectories.tht[node,0:waitStarttht-1] = thtnew[0:waitStarttht-1]
       trajectories.tht[node,waitStarttht:waitStarttht+waitDuration] = thtnew[waitStarttht-1]
       trajectories.tht[node,waitStarttht+waitDuration:self.nSteps] = thtnew[waitStarttht:self.nSteps-waitDuration]
            
       trajectories.phi[node,0:waitStart-1] = phinew[0:waitStart-1]
       trajectories.phi[node,waitStart:waitStart+waitDuration] = phinew[waitStart-1]
       trajectories.phi[node,waitStart+waitDuration:self.nSteps] = phinew[waitStart:self.nSteps-waitDuration] 
            
       trajectories.elbowPositions[node,:] = cobraCenters[node, np.newaxis] + L1[node, np.newaxis] * np.exp(1j * trajectories.tht[node,:])
       trajectories.fiberPositions[node,:] = trajectories.elbowPositions[node,:]  + L2[node, np.newaxis] * np.exp(1j * (trajectories.tht[node,:] + trajectories.phi[node,:]))
       
       return trajectories   
