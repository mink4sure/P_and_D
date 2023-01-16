## PDM
The code in this repository is the code for the project of the course Planning and Decision Making. The goal of the project is to understand the kinematics of a simple robotic model, implement a motion planning algorithm in a simulation environment and validate its performance. In this code a (simplified) motion planning pipeline for a drone is implemented.

It is spilt up into two main sections: The working code, which has an MPC for
local motion planning and a PID for control of the drone and the not so working
code. The not working code is an attempt at full a MPC for a drone, but due to
the complex dynamics, we have hit a little snag. 

**For that code see branch: MPC**

### Local Planning with MPC
This code heavily relies on code provided at [utiasDSL/gym-pybuller-drones](https://github.com/utiasDSL/gym-pybullet-drones). If you want to get this code working, make sure to install the code provided in their repository as mentioned in their README.md.

For the calculation of the solution of the MPC problem, the [CasADi](https://web.casadi.org/) framework is used.

If dependency are succesfully installed this code can be cloned.

To start the simulation, run *fly.py* in the *src/* directory.  

```
$ git clone git@github.com:mink4sure/P_and_D.git
$ cd P_and_D/src/
$ python3 fly.py
```

The constrains and cost functions of the motion planning algorithm are defined in the file *Control.py*

Here a video of te final result:
<img src="https://github.com/mink4sure/P_and_D/blob/main/Drone_gif.gif" width="600" height="338" />

### References
- Panerati, J., Zheng, H., Zhou, S., Xu, J., Prorok, A., & Schoellig, A. P. (2021). Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control. 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 1(1), 1–8. https://doi.org/10.0000/00000
