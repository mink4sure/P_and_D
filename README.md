## PDM
The code in this repository is the code for the project... blablabal

It is spilt up into two main sections: The working code, which has an MPC for
local motion planning and a PID for control of the drone and the not so working
code. The not working code is an attempt at full a MPC for a drone, but due to
the complex dynamics, we have hit a little snag. 

**For the working code see branch: MAIN**

### MPC with a Drone
This code heavily relies on code provided at [utiasDSL/gym-pybuller-drones](https://github.com/utiasDSL/gym-pybullet-drones). If you want to get this code working, make sure to install the code provided in their repository as mentioned in their README.md.

For the calculation of the solution of the MPC problem, the [CasADi](https://web.casadi.org/) framework is used.

To start the simulation, run *fly.py* in the *src/* directory.
