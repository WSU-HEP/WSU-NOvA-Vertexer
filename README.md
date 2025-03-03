# WSU-NOvA-Vertexer
WSU efforts to vertex reconstruction in NOvA


--- 

_Original Author_: Michael Dolce mdolce@fnal.gov

Updated: Feb. 2025 

--- 

### Background
This repository is the home of the NOvA Neutrino Interaction Vertexing project at Wichita State University.

 **The objective of the NOvA vertexing is to improve the NOvA reconstruction algorithms of particle interactions within the detector to improve energy reconstruction**.
 This improved energy reconstruction, in turn, can help to improve a constraint on the neutrino interaction modeling and oscillation parameter constraints.

All simulation used for model training and evaluation is with **NOvA's Production 5.1 campaign**.

### CVN Vertexing 
This project uses a Convolutional Visual Network to identify the true vertex of NOvA neutrino interactions. **We use NOvA's pixel maps (or 'cvnmaps') as the features and the true vertex location as the labels**.

In this project, each vertex coordinate (x, y, z) is trained together, using a ''branch'' model. Each 2D (XZ and YZ) pixel map image is used to learn the (x,y,z) vertex from an interaction. So we have one, single model for the 3D vertex.

The code is divided by NOvA detector: `Far-Detector` and `Near-Detector`. More information can be found within each directory. The latest work haas been done on the Far Detector (FD) interactions. 

This project uses:
- `python 3.11.5`
- `tensorflow 2.15.0.` 


---
