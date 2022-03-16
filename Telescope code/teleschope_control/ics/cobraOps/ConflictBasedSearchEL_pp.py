# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:55:13 2020

@author: rathi
"""

import numpy as np

from . import plotUtils

from .TrajectoryGroup import TrajectoryGroup
from .ConflictTree_pub import Node
#from .ConflictTree_round import Node

import itertools
import multiprocessing
import random
from joblib import Parallel, delayed
import copy

#import multiprocessing
#from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

class ConflictBasedSearch():
    """

    Class used to simulate a PFS observation.

    """

    def __init__(self, bench, targets, trajectorySteps=170, trajectoryStepWidth=50):
        """Constructs a new collision simulator instance.

        Parameters
        ----------
        bench: object
            The PFI bench instance.
        targets: object
            The target group instance.
        nSteps: int, optional
            The total number of steps in the cobra trajectories. Default is
            150.
        stepWidth: int, optional
            The trajectory step width in units of motor steps. Default is 50.

        Returns
        -------
        object
            The collision simulator instance.

        """
        # Save the bench and target group instances
        self.bench = bench
        self.targets = targets
        
        # Save the trajectory parameters
        self.trajectorySteps = trajectorySteps
        self.trajectoryStepWidth = trajectoryStepWidth

        # Check which cobras are assigned to a target
        self.assignedCobras = self.targets.notNull.copy()

        # Define some internal variables that will filled by the run method
        self.finalFiberPositions = None
        self.posSteps = None
        self.negSteps = None
        self.movementDirections = None
        self.movementStrategies = None
        self.trajectories = None
        self.associationCollisions = None
        self.associationEndPointCollisions = None
        self.collisions = None
        self.endPointCollisions = None
        self.nCollisions = None
        self.nEndPointCollisions = None
        self.unassignFiberPos = None
        
    def lenlist(self):
        return self.trajectories.lenlist()
    
    def alllist(self):
        return self.trajectories.alllist()

    def numacq(self):
        return self.trajectories.numacq()
    
    def numstops(self):
        return self.nstops    

    def zeroCollRun(self):
        collAsso = self.run(solveCollisions=True)
        abandonIdx = collAsso[0]
        if len(abandonIdx)>0:
            # abandon cobra and rerun
            self.unassignFiberPos = {cob: self.targets.positions[cob] for cob in abandonIdx}
            self.assignedCobras[abandonIdx] = False
            collAsso = self.run(solveCollisions=True)

    def run(self, solveCollisions=True):
        """Runs the collisions simulator.

        Parameters
        ----------
        solveCollisions: bool, optional
            If True, the simulator will try to solve trajectory collisions,
            changing the movement directions and strategies of the affected
            cobras. Default is True.

        """
        # Calculate the final fiber positions
        self.calculateFinalFiberPositions()

        # Define the theta and phi movement directions
        self.defineMovementDirections()

        # Define the theta and phi movement strategies
        self.defineMovementStrategies()

        
        # Calculate the cobra trajectories
        self.calculateTrajectories()
        # Detect cobra collisions during the trajectory
        #self.detectTrajectoryCollisions()

        #self.replanConflicts()
        
        self.detectTrajectoryCollisions()
        
        random.seed(0)
        count = 0
        lower = True
        self.nstops = 0
        # Solve trajectory collisions if requested
        while solveCollisions and self.nCollisions>0 and count<9:
            self.solveCustomTrajectoryCollisions(lower)
            self.replanConflicts()
            self.detectTrajectoryCollisions()
            #lower = not lower
            lower = bool(random.getrandbits(1))

            count = count+1

            #self.solveTrajectoryCollisions(False)
            #self.solveTrajectoryCollisions(True)
            #self.solveTrajectoryCollisions(False)
        #self.solveConflictsByDelays()
        self.trajectories.recalculatemetrics()
        return self.bench.cobraAssociations[:, self.associationCollisions]
    
    def replanConflicts(self):
        #print(self.collisionPairs)
        #print(self.collisionTimes)
        #totalCost = np.sum(self.alllist())
        self.detectTrajectoryCollisions()
        self.formCobraSubgraphs()
        self.movementStrategies[0,self.collidingCobras] = True
        self.movementStrategies[1,self.collidingCobras] = True
        self.updateTrajectories(self.collidingCobras)
        self.costs = np.array(self.alllist())
        self.allConflicts = []
        addCost = 0
        self.nstops = 0
        #print(self.subGraphs)
        #pool = multiprocessing.Pool(4)
        #zip(*pool.map(self.parbranch, self.subGraphs))
        #processed_list = Parallel(n_jobs=num_cores)(delayed(self.parbranch)(subgraph) for subgraph in self.subGraphs)
        if len(self.subGraphs)>0:
          processed_list = Parallel(n_jobs=num_cores)(delayed(self.parbranch)(subgraph) for subgraph in self.subGraphs)
          #pool = multiprocessing.Pool(16)
          #traj,stops = zip(*pool.map(self.parbranch, self.subGraphs))
          traj, stops = zip(*processed_list)
          for si,subgraph in enumerate(self.subGraphs):
            self.trajectories.tht[subgraph,:] = traj[si].tht[subgraph,:]
            self.trajectories.phi[subgraph,:] = traj[si].phi[subgraph,:]
            self.trajectories.elbowPositions[subgraph,:] = traj[si].elbowPositions[subgraph,:]
            self.trajectories.fiberPositions[subgraph,:] = traj[si].fiberPositions[subgraph,:]
            self.nstops = self.nstops+stops[si]
        #if len(unsolvedSubgraphs>0):
        #  print(unsolvedSubgraphs)
          
    def parbranch(self,subgraph):
      subgraphCost = self.costs[subgraph] 
      addCost = 0
      conflictTreeRoot = Node(self.bench,subgraph,addCost,np.array([]),True,[],None,None,self.trajectories,subgraphCost)
      [newtrajectories,unsolvedSubgraphs,addcost,stops] = self.branchTree(conflictTreeRoot,conflictTreeRoot,self.collisionPairs,self.collisionTimes,subgraph)
      #self.nstops = self.nstops+stops
      return newtrajectories,stops
        
    def branchTree(self,currentNode,rootNode,collisionPairs,collisionTimes,subgraph):   
          indices = []
          unsolvedSubgraphs = []
          subgraphCost = self.costs[subgraph] 
          addCost = 0
          stops = 0
          newtrajectories = copy.deepcopy(self.trajectories)
          for pairindex in range(collisionPairs.shape[1]):
            pair = collisionPairs[:,pairindex]
            if pair[0] in subgraph and pair[1] in subgraph:
                indices.append(pairindex)
          nPairs = len(indices)      
          subgraphPairs = collisionPairs[:,indices]     
          starttimes = np.zeros((subgraphPairs.shape[1],))
          endtimes = np.zeros((subgraphPairs.shape[1],))
          for eleindex in range(nPairs):
             times = np.where(collisionTimes[eleindex]==True)
             starttimes[eleindex] = min(times[0].ravel())
             endtimes[eleindex] = max(times[0].ravel())
             
          children = list(itertools.product([0, 1], repeat=nPairs))
          for child in children:
            conflicts = np.zeros((len(child),5),dtype=np.int)
            for pairindex,pairval in enumerate(child):
              conflicts[pairindex,0] = subgraphPairs[0,pairindex]
              conflicts[pairindex,1] = subgraphPairs[1,pairindex]
              conflicts[pairindex,2] = subgraphPairs[pairval,pairindex]
              conflicts[pairindex,3] = starttimes[pairindex]
              conflicts[pairindex,4] = endtimes[pairindex]
            
            childNode = Node(self.bench,subgraph,-1,conflicts,False,[],currentNode,rootNode,None,subgraphCost)
            currentNode.children.append(childNode)
          currentNode.children.sort(key=lambda x:x.cost)
          for child in currentNode.children:
             cassociationCollisions,ccollisionPairs,ccollisionTimes,ccollidingCobras = self.detectCustomTrajectoryCollisions(child.trajectories)          
             graphcollisions = np.intersect1d(ccollidingCobras,subgraph)
             child.externalConflicts = not (len(graphcollisions)==0)
             subgraphFailure = True
             #print(child.conflicts[0][2])
             if child.internalConflicts==False:
                #print("No internal conflicts") 
                if child.externalConflicts==False: 
                   #print(subgraph,"Success")
                   subgraphFailure = False 
                   newtrajectories.tht[subgraph,:] = child.trajectories.tht[subgraph,:]
                   newtrajectories.phi[subgraph,:] = child.trajectories.phi[subgraph,:]
                   newtrajectories.elbowPositions[subgraph,:] = child.trajectories.elbowPositions[subgraph,:]
                   newtrajectories.fiberPositions[subgraph,:] = child.trajectories.fiberPositions[subgraph,:]
                   #self.allConflicts.extend(child.conflicts)
                   addCost = addCost+child.cost
                   currentNode.successChild = child
                   stops = stops+child.stops
                   break;
                   
          if subgraphFailure==True:
                #print(subgraph,"failure")
                internalFailure = True
                for child in currentNode.children:
                   subgraphnew = subgraph 
                   if child.internalConflicts==False:
                     internalFailure = False  
                     cassociationCollisions,ccollisionPairs,ccollisionTimes,ccollidingCobras = self.detectCustomTrajectoryCollisions(child.trajectories)          
                     for pindex in range(ccollisionPairs.shape[1]):
                        pair = ccollisionPairs[:,pindex] 
                        if pair[0] in subgraph and pair[1] not in subgraph:
                           subgraphnew = np.append(subgraphnew,pair[1])
                        elif pair[1] in subgraph and pair[0] not in subgraph:
                           subgraphnew = np.append(subgraphnew,pair[0])
                     subgraphnew = np.unique(subgraphnew)      
                     newtrajectories,unsolved,childCost,childStops = self.branchTree(child,rootNode,ccollisionPairs,ccollisionTimes,subgraphnew)      
                     stops = stops+childStops
                     #unsolved = []
                if internalFailure==True or len(unsolved)>0:        
                  unsolvedSubgraphs.append(subgraphnew) 
          return newtrajectories,unsolvedSubgraphs,addCost,stops
                        
                        

    def formCobraSubgraphs(self):
        self.visited = np.zeros((self.bench.cobras.nCobras,),dtype=np.bool)
        self.visited[:] = False
        self.subGraphs = []
        for cobraindex,cobra in enumerate(self.collidingCobras):
            if self.visited[cobra]==False:                        
               self.subGraphs.append(self.DFSutil(cobra))
                    
    def DFSutil(self, cobra):
        self.visited[cobra] = True
        connections = np.array([cobra])
        for pairindex in range(self.collisionPairs.shape[1]):
            pair = self.collisionPairs[:,pairindex]
            if pair[0]==cobra and self.visited[pair[1]]==False:
                connections = np.concatenate((connections,self.DFSutil(pair[1])),axis=0)
            elif pair[1]==cobra and self.visited[pair[0]]==False:
                connections = np.concatenate((connections,self.DFSutil(pair[0])),axis=0)
        return connections
    
    def calculateFinalFiberPositions(self):
        """Calculates the cobras final fiber positions.
        """
        # Set the final fiber positions to their associated target positions,
        # leaving unassigned cobras at their home positive positions
        self.finalFiberPositions = self.bench.cobras.home0.copy()
        self.finalFiberPositions[self.assignedCobras] = self.targets.positions[self.assignedCobras]

        # Optimize the unassigned cobra positions to minimize their possible
        # collisions with other cobras
        self.optimizeUnassignedCobraPositions2()
        
    def distancesToLineSegments(self, points, startPoints, endPoints):
        # Translate the points and the line segment end points to the line
        # segment starting points
        translatedPoints = points - startPoints
        translatedEndPoints = endPoints - startPoints

        # Rotate the translated points to have the line segment on the x axis
        rotatedPoints = translatedPoints * np.exp(-1j * np.angle(translatedEndPoints))

        # Define 3 regions for the points: left of the origin, over the line
        # segments, and right of the line segments
        x = rotatedPoints.real
        lineLengths = np.abs(translatedEndPoints)
        region1 = np.where(x <= 0)
        region2 = np.where(np.logical_and(x > 0 , x < lineLengths))
        region3 = np.where(x >= lineLengths)

        # Calculate the minimum distances in each region
        distances = np.empty(rotatedPoints.shape)
        distances[region1] = np.abs(rotatedPoints[region1])
        distances[region2] = np.abs(rotatedPoints[region2].imag)
        distances[region3] = np.abs((rotatedPoints - lineLengths)[region3])

        return distances
        
    def optimizeUnassignedCobraPositions2(self):
        """Finds the unassigned cobras final fiber positions that minimize
        mid-point position of the unassigned cobra other cobras' fiber position.

        """
        # Calculate the cobras elbow positions at their current final fiber
        # positions
        elbowPositions = self.bench.cobras.calculateElbowPositions(self.finalFiberPositions)

        # Get the unassigned cobra indices
        (unassigendCobraIndices,) = np.where(self.assignedCobras == False)

        # Find the optimal position for each unassigned cobra
        for c in unassigendCobraIndices:
            # Get the cobra nearest neighbors
            cobraNeighbors = self.bench.getCobraNeighbors(c)

            # Select only those that are assigned to a target
            cobraNeighbors = cobraNeighbors[self.assignedCobras[cobraNeighbors]]

            # Jump to the next cobra if all the neighbors are unassigned
            if len(cobraNeighbors) == 0:
                continue

            # Calculate all the possible cobra elbow rotations
            
            rotationAngles = np.arange(0, 2 * np.pi, 0.02 * np.pi)
            center = self.bench.cobras.centers[c]
            rotatedElbowPositions = (elbowPositions[c] - center) * np.exp(1j * rotationAngles) + center
            rotatedFiberPositions = (self.finalFiberPositions[c] - center) * np.exp(1j * rotationAngles) + center
            #midPoss = (rotatedElbowPositions + rotatedFiberPositions)/2

            # Obtain the angle that maximizes the closer distance to a neighbor
            if self.unassignFiberPos is not None and c in self.unassignFiberPos:
                finalFiberPos = np.append( self.finalFiberPositions[cobraNeighbors], self.unassignFiberPos[c])
            else:
                finalFiberPos = self.finalFiberPositions[cobraNeighbors]
            #distances = np.abs(finalFiberPos - midPoss[:, np.newaxis])
            distances = self.distancesToLineSegments(finalFiberPos,rotatedElbowPositions[:,None], rotatedFiberPositions[:,None])
            minDistances = np.min(distances, axis=1)
            optimalAngle = rotationAngles[np.argmax(minDistances)]

            # Update the cobra final fiber position
            self.finalFiberPositions[c] = (self.finalFiberPositions[c] - center) * np.exp(1j * optimalAngle) + center    


    def optimizeUnassignedCobraPositions(self):
        """Finds the unassigned cobras final fiber positions that minimize
        their collisions with other cobras.

        """
        # Calculate the cobras elbow positions at their current final fiber
        # positions
        elbowPositions = self.bench.cobras.calculateElbowPositions(self.finalFiberPositions)

        # Get the unassigned cobra indices
        (unassigendCobraIndices,) = np.where(self.assignedCobras == False)

        # Find the optimal position for each unassigned cobra
        for c in unassigendCobraIndices:
            # Get the cobra nearest neighbors
            cobraNeighbors = self.bench.getCobraNeighbors(c)

            # Select only those that are assigned to a target
            cobraNeighbors = cobraNeighbors[self.assignedCobras[cobraNeighbors]]

            # Jump to the next cobra if all the neighbors are unassigned
            if len(cobraNeighbors) == 0:
                continue

            # Calculate all the possible cobra elbow rotations
            rotationAngles = np.arange(0, 2 * np.pi, 0.1 * np.pi)
            center = self.bench.cobras.centers[c]
            rotatedElbowPositions = (elbowPositions[c] - center) * np.exp(1j * rotationAngles) + center

            # Obtain the angle that maximizes the closer distance to a neighbor
            distances = np.abs(self.finalFiberPositions[cobraNeighbors] - rotatedElbowPositions[:, np.newaxis])
            minDistances = np.min(distances, axis=1)
            optimalAngle = rotationAngles[np.argmax(minDistances)]

            # Update the cobra final fiber position
            self.finalFiberPositions[c] = (self.finalFiberPositions[c] - center) * np.exp(1j * optimalAngle) + center

    def defineMovementDirections(self):
        """Defines the theta and phi movement directions that the cobras should
        follow.

        The movement directions are encoded in a boolean array with 2 rows and
        as many columns as cobras in the array:
            - The first row indicates the theta movement direction: positive
            for True values, and negative for False values.
            - The second row indicates the phi movement direction: positive
            for True values, and negative for False values.

        """
        # Get the cobra rotation angles for the starting positive and negative
        # home positions and the final fiber positions
        (posStartTht, posStartPhi) = self.bench.cobras.calculateRotationAngles(self.bench.cobras.home0)
        (negStartTht, negStartPhi) = self.bench.cobras.calculateRotationAngles(self.bench.cobras.home1)
        (finalTht, finalPhi) = self.bench.cobras.calculateRotationAngles(self.finalFiberPositions)

        # Calculate the required theta and phi delta offsets to move from the
        # positive and negative home positions to the final positions
        posDeltaTht = np.mod(finalTht - posStartTht, 2 * np.pi)
        negDeltaTht = -np.mod(negStartTht - finalTht, 2 * np.pi)
        posDeltaPhi = finalPhi - posStartPhi
        negDeltaPhi = finalPhi - negStartPhi

        # Calculate the total number of motor steps required to reach the final
        # positions from the positive and the negative home positions
        (posThtMotorSteps, posPhiMotorSteps) = self.bench.cobras.motorMaps.calculateSteps(posDeltaTht, posStartPhi, posDeltaPhi)
        (negThtMotorSteps, negPhiMotorSteps) = self.bench.cobras.motorMaps.calculateSteps(negDeltaTht, negStartPhi, negDeltaPhi)

        # Calculate the number of trajectory steps required to reach the final
        # positions from the positive and the negative starting positions
        self.posSteps = np.ceil(np.max((posThtMotorSteps, posPhiMotorSteps), axis=0) / self.trajectoryStepWidth).astype("int") + 1
        self.negSteps = np.ceil(np.max((negThtMotorSteps, negPhiMotorSteps), axis=0) / self.trajectoryStepWidth).astype("int") + 1

        # Make sure that at least one of the movements requires less steps than
        # the maximum number of steps allowed
        if np.any(np.min((self.posSteps, self.negSteps), axis=0) > self.trajectorySteps):
            raise Exception("Some cobras cannot reach their assigned targets "
                            "because the trajectorySteps parameter value is "
                            "too low. Please set it to a higher value.")

        # Decide if the cobras should follow a positive theta movement
        # direction:
        #    - The positive movement should require less steps than the
        #    negative movement.
        posThtMovement = self.posSteps < self.negSteps

        # Calculate the phi movement direction
        posPhiMovement = negDeltaPhi > 0
        posPhiMovement[posThtMovement] = posDeltaPhi[posThtMovement] > 0

        # Save the results in a single array
        self.movementDirections = np.vstack((posThtMovement, posPhiMovement))


    def defineMovementStrategies(self):
        """Defines the theta and phi movement strategies that the cobras should
        follow.

        There are only three strategies that really make sense:
            - Early theta and early phi movements, when phi is moving towards
            the cobra center.
            - Early theta and late phi movements, when the theta movement is in
            the positive direction.
            - Late theta and late phi movements, when the theta movement is in
            the negative direction.

        The movement strategies are encoded in a boolean array with 2 rows and
        as many columns as cobras in the array:
            - The first row indicates the theta movement strategy: "early" for
            True values, and "late" for False values.
            - The second row indicates the phi movement strategy: "early" for
            True values, and "late" for False values.

        """
        # By default always do late theta and late phi movements
        thtEarly = np.full(self.bench.cobras.nCobras, False)
        phiEarly = thtEarly.copy()

        # Do early theta movements when moving in the theta positive direction
        thtEarly[self.movementDirections[0]] = True

        # If the cobra is moving towards the center, do the theta and phi
        # movements as early as possible
        towardsTheCenter = np.logical_not(self.movementDirections[1])
        thtEarly[towardsTheCenter] = True
        phiEarly[towardsTheCenter] = True

        # Do early movements for unassigned cobras
        thtEarly[self.assignedCobras == False] = True
        phiEarly[self.assignedCobras == False] = True
        
        #thtEarly[:] = True
        #phiEarly[:] = True
        # Save the results in a single array
        self.movementStrategies = np.vstack((thtEarly, phiEarly))
        
    def updateTrajectories(self,cobraindices):    
        self.trajectories.updateTrajectories(cobraindices,self.movementDirections,self.movementStrategies)

    def calculateTrajectories(self):
        """Calculates the cobra trajectories.

        """
        self.trajectories = TrajectoryGroup(nSteps=self.trajectorySteps,
                                            stepWidth=self.trajectoryStepWidth,
                                            bench=self.bench,
                                            finalFiberPositions=self.finalFiberPositions,
                                            movementDirections=self.movementDirections,
                                            movementStrategies=self.movementStrategies,
                                            avertCollisions=False,assignedCobras=self.assignedCobras)
    
    def detectCustomTrajectoryCollisions(self,customTrajectories):
        """Detects collisions in the cobra trajectories.
        """
        # Detect trajectory collisions between cobra associations
        trajectoryCollisions, distances = customTrajectories.calculateCobraAssociationCollisions()

        # Check which are the cobra associations affected by collisions
        associationCollisions = np.any(trajectoryCollisions, axis=1)
        #associationEndPointCollisions = trajectoryCollisions[:, -1]
        
        collisionPairs = self.bench.cobraAssociations[0:,associationCollisions]
        collisionTimes = trajectoryCollisions[associationCollisions,:]
        collidingCobras = np.unique(self.bench.cobraAssociations[:, associationCollisions])

        return associationCollisions,collisionPairs,collisionTimes,collidingCobras

    def detectTrajectoryCollisions(self):
        """Detects collisions in the cobra trajectories.

        """
        # Detect trajectory collisions between cobra associations
        trajectoryCollisions, self.distances = self.trajectories.calculateCobraAssociationCollisions()

        # Check which are the cobra associations affected by collisions
        self.associationCollisions = np.any(trajectoryCollisions, axis=1)
        self.associationEndPointCollisions = trajectoryCollisions[:, -1]

        # Check which cobras are involved in collisions
        self.collisionPairs = self.bench.cobraAssociations[0:, self.associationCollisions]
        allcollisions = self.collisionPairs.ravel()
        vals, idx_start, count = np.unique(allcollisions, return_counts=True,return_index=True)
        self.multiplecollisions = vals[count>1]
        self.collisionTimes = trajectoryCollisions[self.associationCollisions,:]
        self.collidingCobras = np.unique(self.bench.cobraAssociations[:, self.associationCollisions])
        self.collisions = np.full(self.bench.cobras.nCobras, False)
        self.collisions[self.collidingCobras] = True
        self.nCollisions = np.sum(self.collisions)

        # Check which cobras are involved in end point collisions
        #allcollisions = self.bench.cobraAssociations[:, self.associationEndPointCollisions]
        collidingCobras = np.unique(self.bench.cobraAssociations[:, self.associationEndPointCollisions])
        self.endPointCollisions = np.full(self.bench.cobras.nCobras, False)
        self.endPointCollisions[collidingCobras] = True
        self.nEndPointCollisions = np.sum(self.endPointCollisions)

    def solveCustomTrajectoryCollisions(self, selectLowerIndices):
        """Solves trajectory collisions changing the cobras theta movement
        directions.

        Parameters
        ----------
        selectLowerIndices: bool
            If True, the cobras selected to be changed in the association will
            be the ones with the lower indices values.

        """
        # Get the indices of the cobras involved in a mid point trajectory
        # collision
        associationMidPointCollisions = np.logical_and(self.associationCollisions, self.associationEndPointCollisions == False)
        collidingAssociations = self.bench.cobraAssociations[:, associationMidPointCollisions]

        # Select the indices of the cobras whose movement should be changed
        cobraIndices = np.unique(collidingAssociations[0] if selectLowerIndices else collidingAssociations[1])

        # Make sure that there is at least one cobra to change
        if len(cobraIndices) > 0:
            # Change the cobras theta movement directions
            self.movementDirections[0, cobraIndices] = np.logical_not(self.movementDirections[0, cobraIndices])

            # Make sure that the new movement directions don't require too many
            # steps
            #print(self.posSteps)
            #print(self.negSteps)
            #print(self.posSteps[447])
            #print(self.negSteps[447])
            self.movementDirections[0, self.posSteps > self.trajectories.nSteps] = False
            self.movementDirections[0, self.negSteps > self.trajectories.nSteps] = True

            # Define the theta and phi movement strategies
            self.defineMovementStrategies()

            # Calculate the cobra trajectories
            #print(cobraIndices)
            self.updateTrajectories(cobraIndices)
            #self.calculateTrajectories()
            # Recalculate the cobra collisions during the trajectory
            self.recalculateTrajectoryCollisions(cobraIndices)

    def solveTrajectoryCollisions(self, selectLowerIndices):
        """Solves trajectory collisions changing the cobras theta movement
        directions.

        Parameters
        ----------
        selectLowerIndices: bool
            If True, the cobras selected to be changed in the association will
            be the ones with the lower indices values.

        """
        # Get the indices of the cobras involved in a mid point trajectory
        # collision
        associationMidPointCollisions = np.logical_and(self.associationCollisions, self.associationEndPointCollisions == False)
        collidingAssociations = self.bench.cobraAssociations[:, associationMidPointCollisions]

        # Select the indices of the cobras whose movement should be changed
        cobraIndices = np.unique(collidingAssociations[0] if selectLowerIndices else collidingAssociations[1])

        # Make sure that there is at least one cobra to change
        if len(cobraIndices) > 0:
            # Change the cobras theta movement directions
            self.movementDirections[0, cobraIndices] = np.logical_not(self.movementDirections[0, cobraIndices])

            # Make sure that the new movement directions don't require too many
            # steps
            self.movementDirections[0, self.posSteps > self.trajectories.nSteps] = False
            self.movementDirections[0, self.negSteps > self.trajectories.nSteps] = True

            # Define the theta and phi movement strategies
            self.defineMovementStrategies()

            # Calculate the cobra trajectories
            self.calculateTrajectories()

            # Recalculate the cobra collisions during the trajectory
            self.recalculateTrajectoryCollisions(cobraIndices)


    def recalculateTrajectoryCollisions(self, cobraIndices):
        """Recalculates the trajectory collision information for the given
        cobras.

        Parameters
        ----------
        cobraIndices: object
            A numpy array with the indices of the cobras whose movement has
            changed.

        """
        # Get the cobra associations for the given cobras
        cobraAssociationIndices = np.in1d(self.bench.cobraAssociations[0], cobraIndices)
        cobraAssociationIndices = np.logical_or(cobraAssociationIndices, np.in1d(self.bench.cobraAssociations[1], cobraIndices))

        # Detect trajectory collisions between these cobra associations
        trajectoryCollisions, self.distances = self.trajectories.calculateCobraAssociationCollisions(cobraAssociationIndices)

        # Update the cobra associations affected by collisions
        self.associationCollisions[cobraAssociationIndices] = np.any(trajectoryCollisions, axis=1)
        self.associationEndPointCollisions[cobraAssociationIndices] = trajectoryCollisions[:, -1]

        # Check which cobras are involved in collisions
        self.collidingCobras = np.unique(self.bench.cobraAssociations[:, self.associationCollisions])
        self.collisions[:] = False
        self.collisions[self.collidingCobras] = True
        self.nCollisions = np.sum(self.collisions)

        # Check which cobras are involved in end point collisions
        #allcollisions = self.bench.cobraAssociations[:, self.associationEndPointCollisions]
        collidingCobras = np.unique(self.bench.cobraAssociations[:, self.associationEndPointCollisions])
        self.endPointCollisions[:] = False
        self.endPointCollisions[collidingCobras] = True
        self.nEndPointCollisions = np.sum(self.endPointCollisions)


    def plotResults(self, extraTargets=None, paintFootprints=False):
        """Plots the collision simulator results in a new figure.

        Parameters
        ----------
        extraTargets: object, optional
            Extra targets that should also be plotted in the figure, in
            addition to the targets that were used in the simulation. Default
            is None.
        paintFootprints: bool, optional
            If True, the cobra trajectory footprints will be painted. Default
            is False.

        """
        # Create a new figure
        plotUtils.createNewFigure("Collision simulation results", "x position (mm)", "y position (mm)")

        # Set the axes limits
        limRange = 1.05 * self.bench.radius * np.array([-1, 1])
        xLim = self.bench.center.real + limRange
        yLim = self.bench.center.imag + limRange
        plotUtils.setAxesLimits(xLim, yLim)

        # Draw the cobra patrol areas
        patrolAreaColors = np.full((self.bench.cobras.nCobras, 4), [0.0, 0.0, 1.0, 0.15])
        patrolAreaColors[self.collisions] = [1.0, 0.0, 0.0, 0.3]
        patrolAreaColors[self.endPointCollisions] = [0.0, 1.0, 0.0, 0.5]
        self.bench.cobras.addPatrolAreasToFigure(colors=patrolAreaColors)

        # Draw the cobra links at the final fiber positions
        linkColors = np.full(patrolAreaColors.shape, [0.0, 0.0, 1.0, 0.5])
        linkColors[self.assignedCobras == False] = [1.0, 0.0, 0.0, 0.25]
        self.bench.cobras.addLinksToFigure(self.finalFiberPositions, colors=linkColors)

        # Draw the cobra trajectories and the trajectory footprints of those
        # that have a collision
        footprintColors = np.zeros((self.bench.cobras.nCobras, self.trajectories.nSteps, 4))
        footprintColors[self.collisions, :] = [0.0, 0.0, 1.0, 0.05]
        self.trajectories.addToFigure(paintFootprints=paintFootprints, footprintColors=footprintColors)

        # Draw the targets assigned to the cobras
        self.targets.addToFigure(colors=np.array([1.0, 0.0, 0.0, 1.0]))

        # Draw the extra targets if necessary
        if extraTargets is not None:
            # Draw only those targets that are not part of the simulation
            unusedTargets = np.logical_not(np.in1d(extraTargets.ids, self.targets.ids))
            extraTargets.addToFigure(indices=unusedTargets)


    def animateCobraTrajectory(self, cobraIndex, extraTargets=None, fileName=None):
        """Animates the trajectory of a given cobra and its nearest neighbors.

        Parameters
        ----------
        cobraIndex: int
            The index of the cobra to animate.
        extraTargets: object, optional
            Extra targets that should also be plotted in the figure, in
            addition to the targets that were used in the simulation. Default
            is None.
        fileName: object, optional
            The file name path where a video of the animation should be saved.
            If it is set to None, no video will be saved. Default is None.

        """
        # Extract some useful information
        nCobras = self.bench.cobras.nCobras
        cobraCenters = self.bench.cobras.centers
        rMax = self.bench.cobras.rMax

        # Create a new figure
        plotUtils.createNewFigure("Trajectory animation for cobra " + str(cobraIndex), "x position (mm)", "y position (mm)")

        # Get the cobra neighbors
        cobraNeighbors = self.bench.getCobraNeighbors(cobraIndex)

        # Set the axes limits
        distances = np.abs(cobraCenters[cobraNeighbors] - cobraCenters[cobraIndex])
        limRange = 1.05 * np.max(distances + rMax[cobraNeighbors]) * np.array([-1, 1])
        xLim = cobraCenters[cobraIndex].real + limRange
        yLim = cobraCenters[cobraIndex].imag + limRange
        plotUtils.setAxesLimits(xLim, yLim)

        # Draw the cobra patrol areas
        patrolAreaColors = np.full((nCobras, 4), [0.0, 0.0, 1.0, 0.15])
        patrolAreaColors[self.collisions] = [1.0, 0.0, 0.0, 0.3]
        patrolAreaColors[self.endPointCollisions] = [0.0, 1.0, 0.0, 0.5]
        self.bench.cobras.addPatrolAreasToFigure(colors=patrolAreaColors)

        # Select which cobras should be animated: only those that fall inside
        # the displayed area
        toAnimate = np.full(nCobras, False)
        toAnimate[cobraIndex] = True
        toAnimate[cobraNeighbors] = True
        toAnimate[self.bench.getCobrasNeighbors(cobraNeighbors)] = True

        # Draw the cobras that should not be animated at their final positions
        linkColors = np.full((nCobras, 4), [0.0, 0.0, 1.0, 0.5])
        linkColors[self.assignedCobras == False] = [1.0, 0.0, 0.0, 0.25]
        self.bench.cobras.addLinksToFigure(self.finalFiberPositions, colors=linkColors, indices=np.logical_not(toAnimate))

        # Draw every point in the elbow and fiber trajectories
        plotUtils.addPoints(self.trajectories.elbowPositions.ravel(), s=2, facecolor=[1.0, 1.0, 1.0, 1.0])
        plotUtils.addPoints(self.trajectories.fiberPositions.ravel(), s=2, facecolor=[1.0, 1.0, 1.0, 1.0])

        # Draw the targets assigned to the cobras
        self.targets.addToFigure(colors=np.array([1.0, 0.0, 0.0, 1.0]))

        # Draw the extra targets if necessary
        if extraTargets is not None:
            # Draw only those targets that are not part of the simulation
            unusedTargets = np.logical_not(np.in1d(extraTargets.ids, self.targets.ids))
            extraTargets.addToFigure(indices=unusedTargets)

        # Add the animation
        self.trajectories.addAnimationToFigure(linkColors=linkColors, indices=toAnimate, fileName=fileName)
         
