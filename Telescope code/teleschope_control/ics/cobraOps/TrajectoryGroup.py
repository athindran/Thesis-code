"""

TrajectoryGroup class.

Consult the following papers for more detailed information:

  https://ui.adsabs.harvard.edu/abs/2012SPIE.8450E..17F
  https://ui.adsabs.harvard.edu/abs/2014SPIE.9151E..1YF
  https://ui.adsabs.harvard.edu/abs/2016arXiv160801075T
  https://ui.adsabs.harvard.edu/abs/2018SPIE10707E..28Y
  https://ui.adsabs.harvard.edu/abs/2018SPIE10702E..1CT

"""

import numpy as np
import sys

from . import plotUtils

from .AttributePrinter import AttributePrinter


class TrajectoryGroup(AttributePrinter):
    """

    Class describing the properties of a group of cobra trajectories.

    """

    def __init__(self, nSteps, stepWidth, bench, finalFiberPositions, movementDirections, movementStrategies,avertCollisions,assignedCobras,simulate=False):
        """Constructs a new trajectory group instance.

        Parameters
        ----------
        nSteps: int
            The total number of steps in the trajectory.
        stepWidth: int
            The trajectory step width in units of motor steps.
        bench: object
            The PFI bench instance.
        finalFiberPositions: object
            A complex numpy array with the trajectory final fiber positions for
            each cobra.
        movementDirections: object
            A boolean numpy array with the theta and phi movement directions to
            use. True values indicate that the cobras should move in the
            positive theta or phi directions, while False values indicate that
            the movement should be in the negative direction.
        movementStrategies: object
            A boolean numpy array with the theta and phi movement strategies to
            use. True values indicate that the cobras should move in those
            angles as soon as possible, while False values indicate that the
            angle movement should be as late as possible.

        Returns
        -------
        object
            The trajectory group instance.

        """
        # Save the number of steps in the trajectory and their width
        self.nSteps = nSteps
        self.stepWidth = stepWidth

        # Save the bench instance, the final positions and the movement arrays
        self.bench = bench
        self.finalFiberPositions = finalFiberPositions.copy()
        self.movementDirections = movementDirections.copy()
        self.movementStrategies = movementStrategies.copy()
        self.tht = []
        self.phi = []

        # Calculate the trajectory stating fiber positions
        self.calculateStartingFiberPositions()
        
        # Lookahead horizon length
        if avertCollisions:
            self.tLookAhead = 80;
            self.tBuffer = 0;
        else:
            self.tLookAhead = 0
            self.tBuffer = 0;
        self.assignedCobras = assignedCobras

        # Calculate the cobra trajectories
        if not simulate:
          self.calculateCobraTrajectories()
        else:
          self.simulateCobraTrajectories(avertCollisions)
          #[tht2,phi2] = self.calculateCobraTrajectories()
          #print("Hi")

    def calculateStartingFiberPositions(self):
        """Calculates the trajectories starting fiber positions.

        """
        # Set the start positions according to the specified movement direction
        self.startFiberPositions = self.bench.cobras.home1.copy()
        self.startFiberPositions[self.movementDirections[0]] = self.bench.cobras.home0[self.movementDirections[0]]
    
    def calculateCobraTrajectories(self):
        """Calculates the cobra trajectories using the cobras motor maps.
        """
        # Extract some useful information
        nCobras = self.bench.cobras.nCobras
        cobraCenters = self.bench.cobras.centers
        L1 = self.bench.cobras.L1
        L2 = self.bench.cobras.L2
        motorMaps = self.bench.cobras.motorMaps
        posThtMovement = self.movementDirections[0]
        posPhiMovement = self.movementDirections[1]
        thtEarly = self.movementStrategies[0]
        phiEarly = self.movementStrategies[1]

        # Get the cobra rotation angles for the starting and the final fiber
        # positions
        (startTht, startPhi) = self.bench.cobras.calculateRotationAngles(self.startFiberPositions)
        (finalTht, finalPhi) = self.bench.cobras.calculateRotationAngles(self.finalFiberPositions)

        # Calculate the required theta and phi delta offsets
        deltaTht = -np.mod(startTht - finalTht, 2 * np.pi)
        deltaTht[posThtMovement] = np.mod(finalTht[posThtMovement] - startTht[posThtMovement], 2 * np.pi)
        deltaPhi = finalPhi - startPhi

        # Reassign the final theta positions to be sure that they are
        # consistent with the deltaTht values
        finalTht = startTht + deltaTht

        # Calculate theta and phi angle values along the trajectories
        tht = np.empty((nCobras, self.nSteps))
        phi = np.empty((nCobras, self.nSteps))
        tht[:] = finalTht[:, np.newaxis]
        phi[:] = finalPhi[:, np.newaxis]

        for c in range(nCobras):
            # Jump to the next cobra if the two deltas are zero
            if deltaTht[c] == 0 and deltaPhi[c] == 0:
                continue

            # Get the appropriate theta and phi motor maps
            thtSteps = motorMaps.posThtSteps[c] if posThtMovement[c] else motorMaps.negThtSteps[c]
            phiSteps = motorMaps.posPhiSteps[c] if posPhiMovement[c] else motorMaps.negPhiSteps[c]
            thtOffsets = motorMaps.thtOffsets[c]
            phiOffsets = motorMaps.phiOffsets[c]

            # Get the theta moves from the starting to the final position
            stepLimits = np.interp([0, np.abs(deltaTht[c])], thtOffsets, thtSteps)
            stepMoves = np.concatenate((np.arange(stepLimits[0], stepLimits[1], self.stepWidth), [stepLimits[1]]))
            thtMoves = startTht[c] + np.sign(deltaTht[c]) * np.interp(stepMoves, thtSteps, thtOffsets)

            # Get the phi moves from the starting to the final position
            initOffset = np.pi + startPhi[c] if posPhiMovement[c] else np.abs(startPhi[c])
            stepLimits = np.interp([initOffset, initOffset + np.abs(deltaPhi[c])], phiOffsets, phiSteps)
            stepMoves = np.concatenate((np.arange(stepLimits[0], stepLimits[1], self.stepWidth), [stepLimits[1]]))
            phiMoves = np.interp(stepMoves, phiSteps, phiOffsets)
            phiMoves = phiMoves - np.pi if posPhiMovement[c] else -phiMoves

            # Fill the rotation angles according to the movement strategies
            nThtMoves = len(thtMoves)
            nPhiMoves = len(phiMoves)

            if thtEarly[c]:
                tht[c, :nThtMoves] = thtMoves
            else:
                tht[c, :-nThtMoves] = startTht[c]
                tht[c, -nThtMoves:] = thtMoves

            if phiEarly[c]:
                phi[c, :nPhiMoves] = phiMoves
            else:
                phi[c, :-nPhiMoves] = startPhi[c]
                phi[c, -nPhiMoves:] = phiMoves

        # Calculate the elbow and fiber positions along the trajectory
        self.elbowPositions = cobraCenters[:, np.newaxis] + L1[:, np.newaxis] * np.exp(1j * tht)
        self.fiberPositions = self.elbowPositions + L2[:, np.newaxis] * np.exp(1j * (tht + phi))
        self.finalTht = np.array(finalTht)
        self.finalPhi = np.array(finalPhi)
        num_acq = 0
        total_count = 0
        self.unacquiredcobras = []
        for i in range(self.elbowPositions.shape[0]):
          total_count += 1
          if self.assignedCobras[i] and abs(tht[i,-1]-finalTht[i])<0.2 and abs(phi[i,-1]-finalPhi[i])<0.2:
            num_acq += 1
          elif self.assignedCobras[i]:
            self.unacquiredcobras.append(i)  

        self.num_acq = num_acq

        len_list = []
        all_list = np.zeros((nCobras,))
        for i in range(self.elbowPositions.shape[0]):
          reachedlist = np.where(np.logical_and(abs(tht[i,:]-finalTht[i])<0.01,abs(phi[i,:]-finalPhi[i])<0.01)) 
          if np.size(reachedlist)>0:
            len_list.append(np.min(reachedlist))
            all_list[i] = np.min(reachedlist)

        self.len_list = len_list
        self.all_list = all_list
        self.nstops = 0
        self.tht = tht
        self.phi = phi
        #return [tht,phi]
    
    # This function is used to recalculate trajectories for select cobras with 
    # new movement directions or strateges    
    def updateTrajectories(self,cobraIndices,movementDirections,movementStrategies):
        # Extract some useful information
        nCobras = self.bench.cobras.nCobras
        cobraCenters = self.bench.cobras.centers
        L1 = self.bench.cobras.L1
        L2 = self.bench.cobras.L2
        motorMaps = self.bench.cobras.motorMaps
        self.movementDirections = movementDirections.copy()
        self.movementStrategies = movementStrategies.copy()

        self.calculateStartingFiberPositions()

        posThtMovement = self.movementDirections[0]
        posPhiMovement = self.movementDirections[1]
        thtEarly = self.movementStrategies[0]
        phiEarly = self.movementStrategies[1]
        

        # Get the cobra rotation angles for the starting and the final fiber
        # positions
        (startTht, startPhi) = self.bench.cobras.calculateRotationAngles(self.startFiberPositions)
        (finalTht, finalPhi) = self.bench.cobras.calculateRotationAngles(self.finalFiberPositions)

        # Calculate the required theta and phi delta offsets
        deltaTht = -np.mod(startTht - finalTht, 2 * np.pi)
        deltaTht[posThtMovement] = np.mod(finalTht[posThtMovement] - startTht[posThtMovement], 2 * np.pi)
        deltaPhi = finalPhi - startPhi

        # Reassign the final theta positions to be sure that they are
        # consistent with the deltaTht values
        finalTht = startTht + deltaTht

        # Calculate theta and phi angle values along the trajectories
        #tht[:] = finalTht[:, np.newaxis]
        #phi[:] = finalPhi[:, np.newaxis]

        for c in cobraIndices:
            self.tht[c,:] = finalTht[c]
            self.phi[c,:] = finalPhi[c]
            # Jump to the next cobra if the two deltas are zero
            if deltaTht[c] == 0 and deltaPhi[c] == 0:
                continue

            # Get the appropriate theta and phi motor mapsn
            thtSteps = motorMaps.posThtSteps[c] if posThtMovement[c] else motorMaps.negThtSteps[c]
            phiSteps = motorMaps.posPhiSteps[c] if posPhiMovement[c] else motorMaps.negPhiSteps[c]
            thtOffsets = motorMaps.thtOffsets[c]
            phiOffsets = motorMaps.phiOffsets[c]

            # Get the theta moves from the starting to the final position
            stepLimits = np.interp([0, np.abs(deltaTht[c])], thtOffsets, thtSteps)
            stepMoves = np.concatenate((np.arange(stepLimits[0], stepLimits[1], self.stepWidth), [stepLimits[1]]))
            thtMoves = startTht[c] + np.sign(deltaTht[c]) * np.interp(stepMoves, thtSteps, thtOffsets)

            # Get the phi moves from the starting to the final position
            initOffset = np.pi + startPhi[c] if posPhiMovement[c] else np.abs(startPhi[c])
            stepLimits = np.interp([initOffset, initOffset + np.abs(deltaPhi[c])], phiOffsets, phiSteps)
            stepMoves = np.concatenate((np.arange(stepLimits[0], stepLimits[1], self.stepWidth), [stepLimits[1]]))
            phiMoves = np.interp(stepMoves, phiSteps, phiOffsets)
            phiMoves = phiMoves - np.pi if posPhiMovement[c] else -phiMoves

            # Fill the rotation angles according to the movement strategies
            nThtMoves = len(thtMoves)
            nPhiMoves = len(phiMoves)

            if thtEarly[c]:
                #print(c,len(thtMoves))
                self.tht[c, :nThtMoves] = thtMoves
            else:
                self.tht[c, :-nThtMoves] = startTht[c]
                self.tht[c, -nThtMoves:] = thtMoves

            if phiEarly[c]:
                self.phi[c, :nPhiMoves] = phiMoves
            else:
                self.phi[c, :-nPhiMoves] = startPhi[c]
                self.phi[c, -nPhiMoves:] = phiMoves

        # Calculate the elbow and fiber positions along the trajectory
        self.elbowPositions[cobraIndices,:] = cobraCenters[cobraIndices, np.newaxis] + L1[cobraIndices, np.newaxis] * np.exp(1j * self.tht[cobraIndices,:])
        self.fiberPositions[cobraIndices,:] = self.elbowPositions[cobraIndices,:]  + L2[cobraIndices, np.newaxis] * np.exp(1j * (self.tht[cobraIndices,:] + self.phi[cobraIndices,:]))
        

        num_acq = 0
        total_count = 0
        self.unacquiredcobras = []
        for i in range(self.elbowPositions.shape[0]):
          total_count += 1
          if self.assignedCobras[i] and abs(self.tht[i,-1]-finalTht[i])<0.02 and abs(self.phi[i,-1]-finalPhi[i])<0.02:
            num_acq += 1
          elif self.assignedCobras[i]:
            self.unacquiredcobras.append(i)  

        self.num_acq = num_acq

        len_list = []
        for i in range(self.elbowPositions.shape[0]):
          reachedlist = np.where(np.logical_and(abs(self.tht[i,:]-finalTht[i])<0.02,abs(self.phi[i,:]-finalPhi[i])<0.02)) 
          if np.size(reachedlist)>0:
            len_list.append(np.min(reachedlist))
            self.all_list[i] = np.min(reachedlist)

        self.len_list = len_list
        self.nstops = 0
   
    def recalculatemetrics(self):
        num_acq = 0
        total_count = 0
        self.unacquiredcobras = []
        for i in range(self.elbowPositions.shape[0]):
          total_count += 1
          if self.assignedCobras[i] and abs(np.mod(self.tht[i,-1],2*np.pi)-np.mod(self.finalTht[i],2*np.pi))<0.02 and abs(np.mod(self.phi[i,-1],2*np.pi)-np.mod(self.finalPhi[i],2*np.pi))<0.02:
            num_acq += 1
          elif self.assignedCobras[i]:
            self.unacquiredcobras.append(i)  

        self.num_acq = num_acq

        len_list = []
        for i in range(self.elbowPositions.shape[0]):
          reachedlist = np.where(np.logical_and(abs(np.mod(self.tht[i,:],2*np.pi)-np.mod(self.finalTht[i],2*np.pi))<0.02,abs(np.mod(self.phi[i,:],2*np.pi)-np.mod(self.finalPhi[i],2*np.pi))<0.02)) 
          if np.size(reachedlist)>0:
            len_list.append(np.min(reachedlist))

        self.len_list = len_list
        
    # This function is used to simulate one step at a time
    def simulateCobraTrajectories(self,avertCollisions=False):
        """Calculates the cobra trajectories using the cobras motor maps.

        """
        # Here, I plan the controls, any planning algorithm can go here
        [stepThtMoves,stepPhiMoves,startTht,startPhi,finalTht,finalPhi, deltaTht,deltaPhi] = self.planControls()
        # Extract some useful information
        nCobras = self.bench.cobras.nCobras
        linkRadius = self.bench.cobras.linkRadius

        cobraCenters = self.bench.cobras.centers
        L1 = self.bench.cobras.L1
        L2 = self.bench.cobras.L2
        thtEarly = self.movementStrategies[0]
        phiEarly = self.movementStrategies[1]
        motorMaps = self.bench.cobras.motorMaps
        posThtMovement = self.movementDirections[0]
        posPhiMovement = self.movementDirections[1]

        # Calculate theta and phi angle values along the trajectories
        tht = np.empty((nCobras, self.nSteps+self.tBuffer))
        phi = np.empty((nCobras, self.nSteps+self.tBuffer))
        tht[:] = finalTht[:, np.newaxis]
        phi[:] = finalPhi[:, np.newaxis]
        
        delays = np.zeros((nCobras,),dtype=np.int)
        t = 0;
        stoppedCobras = set()
        #tResetStop = 3
        tResetStop = 3
        self.nstops = 0
        
        while t<self.nSteps+self.tBuffer:
          if avertCollisions and t%tResetStop==0:
            # LookAhead with the Motor Map and Controls
            thtla = np.empty((nCobras, self.tLookAhead))
            phila = np.empty((nCobras, self.tLookAhead))    
            for c in range(nCobras):  
              thtSteps = motorMaps.posThtSteps[c] if posThtMovement[c] else motorMaps.negThtSteps[c]
              phiSteps = motorMaps.posPhiSteps[c] if posPhiMovement[c] else motorMaps.negPhiSteps[c]
              thtOffsets = motorMaps.thtOffsets[c]
              phiOffsets = motorMaps.phiOffsets[c]
              nThtMoves = len(stepThtMoves[c])
              nPhiMoves = len(stepPhiMoves[c])

              if thtEarly[c] and t<nThtMoves+delays[c]:
                # Look ahead required number of timesteps for early moving cobras
                endPoint = min(t+self.tLookAhead-delays[c],nThtMoves)
                thtla[c,0:(endPoint+delays[c]-t)] = self.computeLookAheadTheta(startTht[c],deltaTht[c],stepThtMoves[c][t-delays[c]:endPoint],thtSteps,thtOffsets)
                #thtla[c,(endPoint+delays[c]-t):] = thtla[c,endPoint+delays[c]-t-1]
                thtla[c,(endPoint+delays[c]-t):] = finalTht[c]
              elif not thtEarly[c] and t+self.tLookAhead>self.nSteps-nThtMoves+delays[c] and t<self.nSteps+delays[c]:
                # Look ahead required number of timesteps for late moving cobras
                #if c==740 and t==135:
                #    print("Hello1")
                startmovtime = self.nSteps-nThtMoves+delays[c]
                remnant = t+self.tLookAhead-startmovtime
                startPoint = self.tLookAhead-remnant
                if startPoint>0:
                  thtla[c,0:startPoint] = startTht[c]
                else:
                  startPoint = 0
                  remnant = self.tLookAhead
                
                startPointarr = max(0,t-startmovtime)
                endPointarr = min(startPointarr+remnant,nThtMoves)
                #if t==111 and c==501:
                #  print(t,c)
                thtla[c,startPoint:startPoint+(endPointarr-startPointarr)] = self.computeLookAheadTheta(startTht[c],deltaTht[c],stepThtMoves[c][startPointarr:endPointarr],thtSteps,thtOffsets)
                #thtla[c,(endPoint-nThtMoves+self.nSteps+delays[c]-t):] = thtla[c,(endPoint-nThtMoves+self.nSteps+delays[c]-t)-1]
                thtla[c,startPoint+(endPointarr-startPointarr):] = finalTht[c]
              # This clause implies an early moving cobra has reached the destination
              elif thtEarly[c] and t>=nThtMoves+delays[c]:
                thtla[c,0:self.tLookAhead] = finalTht[c]
              # This clause implies an late moving cobra has not started  
              elif not thtEarly[c] and t<=self.nSteps-nThtMoves+delays[c]-self.tLookAhead:
                thtla[c,0:self.tLookAhead] = startTht[c]
              # This clause implies an late moving cobra has reached the destination
              elif not thtEarly[c] and t>=self.nSteps+delays[c]-1:
                thtla[c,0:self.tLookAhead] = finalTht[c]
                
              if phiEarly[c] and t<nPhiMoves+delays[c]:
                # Look ahead required number of timesteps for early moving cobras  
                endPoint = min(t+self.tLookAhead-delays[c],nPhiMoves) 
                phila[c,0:(endPoint+delays[c]-t)] = self.computeLookAheadPhi(stepPhiMoves[c][t-delays[c]:endPoint],phiSteps,phiOffsets,posPhiMovement[c])
                #phila[c,(endPoint+delays[c]-t):] = phila[c,endPoint+delays[c]-t-1]
                phila[c,(endPoint+delays[c]-t):] = finalPhi[c]
              elif not phiEarly[c] and t+self.tLookAhead>self.nSteps-nPhiMoves+delays[c] and t<self.nSteps+delays[c]:
                # Look ahead required number of timesteps for late moving cobras
                startmovtime = self.nSteps-nPhiMoves+delays[c]
                remnant = t+self.tLookAhead-startmovtime
                startPoint = self.tLookAhead-remnant
                if startPoint>0:
                  phila[c,0:startPoint] = startPhi[c]
                else:
                  startPoint = 0
                  remnant = self.tLookAhead
                startPointarr = max(0,t-startmovtime)
                endPointarr = min(startPointarr+remnant,nPhiMoves)
                phila[c,startPoint:startPoint+(endPointarr-startPointarr)] = self.computeLookAheadPhi(stepPhiMoves[c][startPointarr:endPointarr],phiSteps,phiOffsets,posPhiMovement[c])
                #phila[c,(endPoint-nPhiMoves+self.nSteps+delays[c]-t):] = phila[c,(endPoint-nPhiMoves+self.nSteps+delays[c]-t)-1]
                phila[c,startPoint+(endPointarr-startPointarr):] = finalPhi[c]
              # This clause implies an early moving cobra has reached the destination  
              elif phiEarly[c] and t>=nPhiMoves+delays[c]:
                phila[c,0:self.tLookAhead] = finalPhi[c]
              # This clause implies an late moving cobra has not started    
              elif not phiEarly[c] and t<=self.nSteps-nPhiMoves+delays[c]-self.tLookAhead:
                phila[c,0:self.tLookAhead] = startPhi[c]
              # This clause implies an late moving cobra has reached the destination  
              elif not phiEarly[c] and t>=self.nSteps+delays[c]-1:
                phila[c,0:self.tLookAhead] = finalPhi[c]
                
            # Calculate elbow positions and fiber positions for the lookahead    
            elbowPositionsla = cobraCenters[:, np.newaxis] + L1[:, np.newaxis] * np.exp(1j * thtla)
            fiberPositionsla = elbowPositionsla + L2[:, np.newaxis] * np.exp(1j * (thtla + phila))
            elbowPositionslastart = elbowPositionsla[:,0]
            fiberPositionslastart = fiberPositionsla[:,0]
            elbowPositionslaend = elbowPositionsla[:,-1]
            fiberPositionslaend = fiberPositionsla[:,-1]
            
            # Detect collisions in the lookahead
            self.detectCustomTrajectoryCollisions(elbowPositionsla,fiberPositionsla)
          
            stoppedCobrasnew = set()
            # self.stopCobras0 contains the cobras that have collided. 
            # Look inside the detectCustiomTrajectoryCollisions
            for cindex,cstop in enumerate(self.stopCobras0):
              # Find start and end elbow and fiberpositions for lookahead
              #print(t,self.stopCobras0,self.stopCobras1)
              #print(thtla[self.stopCobras0[0],:])
              #print(thtla[self.stopCobras1[0],:])
              #print(phila[self.stopCobras0[0],:])
              #print(phila[self.stopCobras1[0],:])
              #print(thtEarly[self.stopCobras1[0]])
              #sys.exit()
              startposfiber1 = np.repeat(fiberPositionslastart[self.stopCobras1[cindex]],self.tLookAhead,axis=0)
              startposelbow1 = np.repeat(elbowPositionslastart[self.stopCobras1[cindex]],self.tLookAhead,axis=0)
              startposfiber0 = np.repeat(fiberPositionslastart[self.stopCobras0[cindex]],self.tLookAhead,axis=0)
              startposelbow0 = np.repeat(elbowPositionslastart[self.stopCobras0[cindex]],self.tLookAhead,axis=0)
              
              endposfiber1 = np.repeat(fiberPositionslaend[self.stopCobras1[cindex]],self.tLookAhead,axis=0)
              endposelbow1 = np.repeat(elbowPositionslaend[self.stopCobras1[cindex]],self.tLookAhead,axis=0)
              endposfiber0 = np.repeat(fiberPositionslaend[self.stopCobras0[cindex]],self.tLookAhead,axis=0)
              endposelbow0 = np.repeat(elbowPositionslaend[self.stopCobras0[cindex]],self.tLookAhead,axis=0)
              
              # Would there be a collision if I stopped either of the two cobras
              distance1 = self.bench.distancesBetweenLineSegments(fiberPositionsla[self.stopCobras0[cindex],:], elbowPositionsla[self.stopCobras0[cindex],:],startposfiber1,startposelbow1)
              distance0 = self.bench.distancesBetweenLineSegments(fiberPositionsla[self.stopCobras1[cindex],:], elbowPositionsla[self.stopCobras1[cindex],:],startposfiber0,startposelbow0)
             
              distance3 = self.bench.distancesBetweenLineSegments(fiberPositionsla[self.stopCobras0[cindex],:], elbowPositionsla[self.stopCobras0[cindex],:],endposfiber1,endposelbow1)
              distance2 = self.bench.distancesBetweenLineSegments(fiberPositionsla[self.stopCobras1[cindex],:], elbowPositionsla[self.stopCobras1[cindex],:],endposfiber0,endposelbow0)   
              coll1 = any(distance1<(linkRadius[self.stopCobras0[cindex], np.newaxis] + linkRadius[self.stopCobras1[cindex], np.newaxis]))
              coll0 = any(distance0<(linkRadius[self.stopCobras0[cindex], np.newaxis] + linkRadius[self.stopCobras1[cindex], np.newaxis]))
              coll3 = any(distance3<(linkRadius[self.stopCobras0[cindex], np.newaxis] + linkRadius[self.stopCobras1[cindex], np.newaxis]))
              coll2 = any(distance2<(linkRadius[self.stopCobras0[cindex], np.newaxis] + linkRadius[self.stopCobras1[cindex], np.newaxis]))
              
              # If stopping one cobra is not enough, stop both the cobras
              if coll1 and coll0:
                 if self.stopCobras0[cindex] not in stoppedCobras and t>1:
                     self.nstops = self.nstops+1
                 if self.stopCobras1[cindex] not in stoppedCobras and t>1:
                     self.nstops = self.nstops+1
                 stoppedCobrasnew.add(self.stopCobras0[cindex])
                 stoppedCobrasnew.add(self.stopCobras1[cindex])
              # Stopping cobra 1 prevents the collisions. So stop cobra 1 in pair (0,1)   
              elif coll0 and not coll1:
                 stoppedCobrasnew.add(self.stopCobras1[cindex])
                 if self.stopCobras1[cindex] not in stoppedCobras and t>1:
                     self.nstops = self.nstops+1
              # Stopping cobra 0 prevents the collisions. So stop cobra 0 in pair (0,1)
              elif coll1 and not coll0:
                 if self.stopCobras0[cindex] not in stoppedCobras and t>1:
                     self.nstops = self.nstops+1 
                 stoppedCobrasnew.add(self.stopCobras0[cindex])
              # Stopping either cobra prevents the collision.   
              elif not coll1 and not coll0:
                 # Cobra 0 in the endpoint is not in the way of Cobra 1
                 if coll3 and not coll2:
                   if self.stopCobras1[cindex] not in stoppedCobras and t>1:
                     self.nstops = self.nstops+1
                   stoppedCobrasnew.add(self.stopCobras1[cindex])
                 # Cobra 1 in the endpoint is not in the way of Cobra 0  
                 elif coll2 and not coll3:
                   if self.stopCobras0[cindex] not in stoppedCobras and t>1:
                     self.nstops = self.nstops+1
                   stoppedCobrasnew.add(self.stopCobras0[cindex]) 
                 # Stop the cobra already stopped  
                 elif self.stopCobras1[cindex] in stoppedCobras:
                   if self.stopCobras1[cindex] not in stoppedCobras and t>1:
                      self.nstops = self.nstops+1 
                   stoppedCobrasnew.add(self.stopCobras1[cindex])
                 else:
                   if self.stopCobras0[cindex] not in stoppedCobras and t>1:
                      self.nstops = self.nstops+1 
                   stoppedCobrasnew.add(self.stopCobras0[cindex])

              #if self.stopCobras0[cindex] in stoppedCobras and self.stopCobras1[cindex] not in stoppedCobras:
              #  stoppedCobrasnew.add(self.stopCobras0[cindex])
              #elif self.stopCobras0[cindex] not in stoppedCobras and self.stopCobras1[cindex] in stoppedCobras:
              #  stoppedCobrasnew.add(self.stopCobras1[cindex])   
              #else:
              #  self.nstops = self.nstops+1  
              #  #if np.random.rand()>0.5:
              #  closeness0 = abs(finalPhi[self.stopCobras0[cindex]]-phila[self.stopCobras0[cindex],0])+abs(finalTht[self.stopCobras0[cindex]]-thtla[self.stopCobras0[cindex],0])
              #  closeness1 = abs(finalPhi[self.stopCobras1[cindex]]-phila[self.stopCobras1[cindex],0])+abs(finalTht[self.stopCobras1[cindex]]-thtla[self.stopCobras1[cindex],0])
              #  if closeness1>closeness0 or (self.assignedCobras[self.stopCobras1[cindex]] and not self.assignedCobras[self.stopCobras0[cindex]]):
              #   stoppedCobrasnew.add(self.stopCobras1[cindex])
              #  else:
              #   stoppedCobrasnew.add(self.stopCobras0[cindex])             
            stoppedCobras = stoppedCobrasnew
            #stoppedCobras = set()   
            for sc in stoppedCobras:
              delays[sc] += tResetStop
          # Step forward in time  
          for c in range(nCobras):
            # Jump to the next cobra if the two deltas are zero
            if stepThtMoves[c]==[] and stepPhiMoves[c]==[]:
                continue

            # Get the appropriate theta and phi motor maps
            thtSteps = motorMaps.posThtSteps[c] if posThtMovement[c] else motorMaps.negThtSteps[c]
            phiSteps = motorMaps.posPhiSteps[c] if posPhiMovement[c] else motorMaps.negPhiSteps[c]
            thtOffsets = motorMaps.thtOffsets[c]
            phiOffsets = motorMaps.phiOffsets[c]
            nThtMoves = len(stepThtMoves[c])
            nPhiMoves = len(stepPhiMoves[c])

            if thtEarly[c] and t<nThtMoves+delays[c]:
               # Control I am applying
               if c not in stoppedCobras:
                 controlTht = stepThtMoves[c][t-delays[c]]
                 # Apply control and move forward in time
                 tht[c,t] = startTht[c]+np.sign(deltaTht[c])*np.interp(controlTht, thtSteps, thtOffsets)
               elif t>0:
                 tht[c,t] = tht[c,t-1]
               else:
                 tht[c,t] = startTht[c]  
            elif not thtEarly[c] and t>self.nSteps-nThtMoves+delays[c] and t<self.nSteps+delays[c]-1:
               if c not in stoppedCobras: 
                 # Control I am applying 
                 controlTht = stepThtMoves[c][t-(self.nSteps-nThtMoves)-delays[c]] 
                 # Apply control and move forward in time
                 tht[c,t] = startTht[c]+np.sign(deltaTht[c])*np.interp(controlTht, thtSteps, thtOffsets)
               elif t>0:
                 tht[c,t] = tht[c,t-1]
               else:
                 tht[c,t] = startTht[c]  
            elif thtEarly[c] and t>=nThtMoves+delays[c]:
               tht[c,t] = finalTht[c]
            elif not thtEarly[c] and t<=self.nSteps-nThtMoves+delays[c]:
               tht[c,t] = startTht[c]
            elif not thtEarly[c] and t>=self.nSteps+delays[c]-1:
               tht[c,t] = finalTht[c] 

            if phiEarly[c] and t<nPhiMoves+delays[c]:
               if c not in stoppedCobras: 
                 # Control I am applying 
                 controlPhi = stepPhiMoves[c][t-delays[c]] 
                 # Apply control and move forward in time
                 phi[c,t] = np.interp(controlPhi, phiSteps, phiOffsets)
                 phi[c,t] = phi[c,t] - np.pi if posPhiMovement[c] else -phi[c,t]
               elif t>0:
                 phi[c,t] = phi[c,t-1]
               else:
                 phi[c,t] = startPhi[c]
                 
            elif not phiEarly[c] and t>self.nSteps-nPhiMoves+delays[c] and t<self.nSteps+delays[c]-1:
               if c not in stoppedCobras: 
                 # Control I am applying 
                 controlPhi = stepPhiMoves[c][t-(self.nSteps-nPhiMoves)-delays[c]] 
                 # Apply control and move forward in time
                 phi[c,t] = np.interp(controlPhi, phiSteps, phiOffsets)
                 phi[c,t] = phi[c,t] - np.pi if posPhiMovement[c] else -phi[c,t]
               elif t>0:
                 phi[c,t] = phi[c,t-1]
               else:
                 phi[c,t] = startPhi[c]
            elif phiEarly[c] and t>=nPhiMoves+delays[c]:
               phi[c,t] = finalPhi[c]
               #phi[c,t] = phi[c,t] - np.pi if posPhiMovement[c] else -phi[c,t]
            elif not phiEarly[c] and t<=self.nSteps-nPhiMoves+delays[c]:
               phi[c,t] = startPhi[c]   
               #phi[c,t] = phi[c,t] - np.pi if posPhiMovement[c] else -phi[c,t]
            elif not phiEarly[c] and t>=self.nSteps+delays[c]-1:
               phi[c,t] = finalPhi[c]
          t = t+1
        # Calculate the elbow and fiber positions along the trajectory
        self.elbowPositions = cobraCenters[:, np.newaxis] + L1[:, np.newaxis] * np.exp(1j * tht)
        self.fiberPositions = self.elbowPositions + L2[:, np.newaxis] * np.exp(1j * (tht + phi))
        
        #if avertCollisions:
        #  print(tht[738,135:])
        #  print(phi[738,135:])
        #  print(tht[740,135:])
        #  print(phi[740,135:])
        num_acq = 0
        total_count = 0
        self.unacquiredcobras = []
        self.tht = tht
        self.phi = phi
        self.finalTht = np.array(finalTht)
        self.finalPhi = np.array(finalPhi)
        #print(np.sum(self.assignedCobras))
        for i in range(self.elbowPositions.shape[0]):
          total_count += 1
          if self.assignedCobras[i] and abs(self.tht[i,-1]-finalTht[i])<0.2 and abs(self.phi[i,-1]-finalPhi[i])<0.2:
            num_acq += 1
          elif self.assignedCobras[i]:
            self.unacquiredcobras.append(i)  

        self.num_acq = num_acq

        len_list = []
        for i in range(self.elbowPositions.shape[0]):
          reachedlist = np.where(np.logical_and(abs(self.tht[i,:]-finalTht[i])<0.2,abs(self.phi[i,:]-finalPhi[i])<0.2)) 
          if np.size(reachedlist)>0:
            len_list.append(np.min(reachedlist))
        self.len_list = len_list
        
    def lenlist(self):
        return self.len_list
    
    def alllist(self):
        return self.all_list
    
    def numacq(self):
        return self.num_acq, self.unacquiredcobras
    
    def numstops(self):
        return self.nstops

    def computeLookAheadTheta(self,start,delta,controls,thtSteps,thtOffsets):
        thtla = start+np.sign(delta)*np.interp(controls, thtSteps, thtOffsets)
        return thtla
    
    def computeLookAheadPhi(self,controls,phiSteps,phiOffsets, posMove):
        phila = np.interp(controls,phiSteps,phiOffsets)
        phila = phila - np.pi if posMove else -phila
        return phila
          
    def detectCustomTrajectoryCollisions(self,elbowPositions,fiberPositions):
        """Detects collisions in the cobra trajectories.

        """
        # Detect trajectory collisions between cobra associations
        trajectoryCollisions, self.distances = self.calculateCustomCobraAssociationCollisions(elbowPositions,fiberPositions)
        #print(np.shape(trajectoryCollisions))
        
        # Check which are the cobra associations affected by collisions
        self.associationCollisions = np.any(trajectoryCollisions, axis=1)
        self.associationEndPointCollisions = trajectoryCollisions[:, -1]

        # Check which cobras are involved in collisions
        self.stopCobras0 = self.bench.cobraAssociations[0,self.associationCollisions]
        self.stopCobras1 = self.bench.cobraAssociations[1,self.associationCollisions]
        self.collidingCobrasAll = np.unique(self.bench.cobraAssociations[:, self.associationCollisions])
        self.collisions = np.full(self.bench.cobras.nCobras, False)
        self.collisions[self.collidingCobrasAll] = True
        self.nCollisions = np.sum(self.collisions)

        # Check which cobras are involved in end point collisions
        self.collidingCobrasEP = np.unique(self.bench.cobraAssociations[:, self.associationEndPointCollisions])
        self.endPointCollisions = np.full(self.bench.cobras.nCobras, False)
        self.endPointCollisions[self.collidingCobrasEP] = True
        self.nEndPointCollisions = np.sum(self.endPointCollisions)    
        
    def calculateCustomCobraAssociationCollisions(self, elbowPositions, fiberPositions, associationIndices=None):
        """Calculates which cobra associations are involved in a collision for
        each step in the trajectory.

        Parameters
        ----------
        associationIndices: object, optional
            A numpy array with the cobra associations indices to use. If it is
            set to None, all the cobra associations will be used. Default is
            None.

        Returns
        -------
        tuple
            A python tuple with a boolean numpy array indicating which cobra
            associations are involved in a collision for each step in the
            trajectory and a double numpy array with the cobra association
            distances along the trajectories.

        """
        # Extract some useful information
        cobraAssociations = self.bench.cobraAssociations
        linkRadius = self.bench.cobras.linkRadius

        # Select a subset of the cobra associations if necessary
        if associationIndices is not None:
            cobraAssociations = cobraAssociations[:, associationIndices]

        # Calculate the distances between the cobras links for each step in the
        # trajectory
        startPoints1 = fiberPositions[cobraAssociations[0]].ravel()
        endPoints1 = elbowPositions[cobraAssociations[0]].ravel()
        startPoints2 = fiberPositions[cobraAssociations[1]].ravel()
        endPoints2 = elbowPositions[cobraAssociations[1]].ravel()
        distances = self.bench.distancesBetweenLineSegments(startPoints1, endPoints1, startPoints2, endPoints2)

        # Reshape the distances array
        distances = distances.reshape((len(cobraAssociations[0]), np.shape(elbowPositions)[1]))

        # Return the cobra association collisions along the trajectory and the
        # distances array
        return distances < (linkRadius[cobraAssociations[0], np.newaxis] + linkRadius[cobraAssociations[1], np.newaxis]), distances
    
        
    
    def planControls(self):
        nCobras = self.bench.cobras.nCobras

        motorMaps = self.bench.cobras.motorMaps
        posThtMovement = self.movementDirections[0]
        posPhiMovement = self.movementDirections[1]

        # Get the cobra rotation angles for the starting and the final fiber
        # positions
        (startTht, startPhi) = self.bench.cobras.calculateRotationAngles(self.startFiberPositions)
        (finalTht, finalPhi) = self.bench.cobras.calculateRotationAngles(self.finalFiberPositions)

        # Calculate the required theta and phi delta offsets
        deltaTht = -np.mod(startTht - finalTht, 2 * np.pi)
        deltaTht[posThtMovement] = np.mod(finalTht[posThtMovement] - startTht[posThtMovement], 2 * np.pi)
        deltaPhi = finalPhi - startPhi

        # Reassign the final theta positions to be sure that they are
        # consistent with the deltaTht values
        finalTht = startTht + deltaTht
        stepThtMoves = [[]]*nCobras
        stepPhiMoves = [[]]*nCobras
        
        for c in range(nCobras):
          # Jump to the next cobra if the two deltas are zero
          if deltaTht[c] == 0 and deltaPhi[c] == 0:
            continue
          # Get the appropriate theta and phi motor maps
          thtSteps = motorMaps.posThtSteps[c] if posThtMovement[c] else motorMaps.negThtSteps[c]
          phiSteps = motorMaps.posPhiSteps[c] if posPhiMovement[c] else motorMaps.negPhiSteps[c]
          thtOffsets = motorMaps.thtOffsets[c]
          phiOffsets = motorMaps.phiOffsets[c]

          # Get the theta moves from the starting to the final position
          stepLimits = np.interp([0, np.abs(deltaTht[c])], thtOffsets, thtSteps)
          stepThtMoves[c] = np.concatenate((np.arange(stepLimits[0], stepLimits[1], self.stepWidth), [stepLimits[1]]))

          # Get the phi moves from the starting to the final position
          initOffset = np.pi + startPhi[c] if posPhiMovement[c] else np.abs(startPhi[c])
          stepLimits = np.interp([initOffset, initOffset + np.abs(deltaPhi[c])], phiOffsets, phiSteps)
          stepPhiMoves[c] = np.concatenate((np.arange(stepLimits[0], stepLimits[1], self.stepWidth), [stepLimits[1]]))
        return stepThtMoves,stepPhiMoves, startTht, startPhi, finalTht, finalPhi, deltaTht, deltaPhi

    def calculateCobraAssociationCollisions(self, associationIndices=None):
        """Calculates which cobra associations are involved in a collision for
        each step in the trajectory.

        Parameters
        ----------
        associationIndices: object, optional
            A numpy array with the cobra associations indices to use. If it is
            set to None, all the cobra associations will be used. Default is
            None.

        Returns
        -------
        tuple
            A python tuple with a boolean numpy array indicating which cobra
            associations are involved in a collision for each step in the
            trajectory and a double numpy array with the cobra association
            distances along the trajectories.

        """
        # Extract some useful information
        cobraAssociations = self.bench.cobraAssociations
        linkRadius = self.bench.cobras.linkRadius

        # Select a subset of the cobra associations if necessary
        if associationIndices is not None:
            cobraAssociations = cobraAssociations[:, associationIndices]

        # Calculate the distances between the cobras links for each step in the
        # trajectory
        startPoints1 = self.fiberPositions[cobraAssociations[0]].ravel()
        endPoints1 = self.elbowPositions[cobraAssociations[0]].ravel()
        startPoints2 = self.fiberPositions[cobraAssociations[1]].ravel()
        endPoints2 = self.elbowPositions[cobraAssociations[1]].ravel()
        distances = self.bench.distancesBetweenLineSegments(startPoints1, endPoints1, startPoints2, endPoints2)

        # Reshape the distances array
        distances = distances.reshape((len(cobraAssociations[0]), self.nSteps+self.tBuffer))

        # Return the cobra association collisions along the trajectory and the
        # distances array
        return distances < (linkRadius[cobraAssociations[0], np.newaxis] + linkRadius[cobraAssociations[1], np.newaxis]), distances


    def addToFigure(self, colors=np.array([0.4, 0.4, 0.4, 1.0]), indices=None, paintFootprints=False, footprintColors=np.array([0.0, 0.0, 1.0, 0.05])):
        """Draws the cobra trajectories on top of an existing figure.

        Parameters
        ----------
        colors: object, optional
            The trajectory colors. Default is dark grey.
        indices: object, optional
            A numpy array with the cobra trajectory indices to use. If it is
            set to None, all the trajectories will be used. Default is None.
        paintFootprints: bool, optional
            If True, the cobra footprints will be painted. Default is False.
        footprintColors: object, optional
            The cobra footprints colors. Default is very light blue.

        """
        # Extract some useful information
        elbowPositions = self.elbowPositions
        fiberPositions = self.fiberPositions
        linkRadius = self.bench.cobras.linkRadius

        # Select a subset of the trajectories if necessary
        if indices is not None:
            elbowPositions = elbowPositions[indices]
            fiberPositions = fiberPositions[indices]
            linkRadius = linkRadius[indices]

            if colors.ndim >= 2:
                colors = colors[indices]

            if footprintColors.ndim >= 2:
                footprintColors = footprintColors[indices]

        # Plot the elbow and fiber trajectories as continuous lines
        plotUtils.addTrajectories(np.vstack((elbowPositions, fiberPositions)), color=np.vstack((colors, colors)), linewidth=1)

        # Paint the cobra trajectory footprints if necessary
        if paintFootprints:
            # Calculate the line thicknesses
            thiknesses = np.empty(elbowPositions.shape)
            thiknesses[:] = linkRadius[:, np.newaxis]

            # Only use the elbow and fiber positions where the cobra is moving
            isMoving = np.empty(elbowPositions.shape, dtype="bool")
            isMoving[:, :-1] = (fiberPositions[:, 1:] - fiberPositions[:, :-1]) != 0
            isMoving[:, -1] = isMoving[:, -2]
            elbowPositions = elbowPositions[isMoving]
            fiberPositions = fiberPositions[isMoving]
            thiknesses = thiknesses[isMoving]

            # Update the colors if necessary
            if footprintColors.ndim > 2 and footprintColors.shape[:2] == isMoving.shape:
                # Set the colors for the moving positions
                footprintColors = footprintColors[isMoving]

                # Only use positions where the alpha color is not exactly zero
                visible = footprintColors[:, 3] != 0
                elbowPositions = elbowPositions[visible]
                fiberPositions = fiberPositions[visible]
                thiknesses = thiknesses[visible]
                footprintColors = footprintColors[visible]

            # Represent the trajectory footprint as a combination of thick
            # lines
            plotUtils.addThickLines(fiberPositions, elbowPositions, thiknesses, facecolor=footprintColors)


    def addAnimationToFigure(self, colors=np.array([0.4, 0.4, 0.4, 1.0]), linkColors=np.array([0.0, 0.0, 1.0, 0.5]), indices=None, fileName=None):
        """Animates the cobra trajectories on top of an existing figure.

        Parameters
        ----------
        colors: object, optional
            The trajectory colors. Default is dark grey.
        linkColors: object, optional
            The cobra link colors. Default is light blue.
        indices: object, optional
            A numpy array with the cobra trajectory indices to animate. If it
            is set to None, all the trajectories will be animated. Default is
            None.
        fileName: object, optional
            The file name path where a video of the animation should be saved.
            If it is set to None, no video will be saved. Default is None.

        """
        # Extract some useful information
        elbowPositions = self.elbowPositions
        fiberPositions = self.fiberPositions
        cobraCenters = self.bench.cobras.centers
        linkRadius = self.bench.cobras.linkRadius

        # Select a subset of the trajectories if necessary
        if indices is not None:
            elbowPositions = elbowPositions[indices]
            fiberPositions = fiberPositions[indices]
            cobraCenters = cobraCenters[indices]
            linkRadius = linkRadius[indices]

            if colors.ndim >= 2:
                colors = colors[indices]

            if linkColors.ndim >= 2:
                linkColors = linkColors[indices]

        # Define the animation update function
        lineCollection = [None]
        thickLineCollection = [None]
        trajectoryCollection = [None]

        def update(frame):
            # Remove the cobras line collections painted in the previous step
            if lineCollection[0] is not None:
                plotUtils.plt.gca().collections.remove(lineCollection[0])
                plotUtils.plt.gca().collections.remove(thickLineCollection[0])
                plotUtils.plt.gca().collections.remove(trajectoryCollection[0])

            # Draw the cobras using a combination of thin and thick lines
            lineCollection[0] = plotUtils.addLines(cobraCenters, elbowPositions[:, frame], edgecolor=linkColors, linewidths=2)
            thickLineCollection[0] = plotUtils.addThickLines(elbowPositions[:, frame], fiberPositions[:, frame], linkRadius, facecolors=linkColors)

            # Draw also their line trajectories
            combinedTrajectories = np.vstack((elbowPositions[:, :frame + 1], fiberPositions[:, :frame + 1]))
            trajectoryCollection[0] = plotUtils.addTrajectories(combinedTrajectories, color=np.vstack((colors, colors)), linewidth=1)

        # Add the animation to the current figure
        plotUtils.addAnimation(update, elbowPositions.shape[1], fileName=fileName)
