import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

# For MPC control
import do_mpc
import cvxpy as cp

class MPC(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([ [0, -1,  -1], [+1, 0, 1], [0,  1,  -1], [-1, 0, 1] ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """
        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """

        N = 20      # MPC Horizon
        N_x = 13    # Size of state vector
        N_u = 4     # Size of control vector

        weight_input = 0.2*np.eye(N_u)    # Weight on the input
        weight_tracking = 1.0*np.eye(N_x) # Weight on the tracking state

        cost = 0.
        constraints = []
    
        # Create the optimization variables
        x = cp.Variable((2, N + 1)) # cp.Variable((dim_1, dim_2))
        u = cp.Variable((1, N))

        # Converting current rotation Quat to een Rotations matrix
        cur_rpy = np.zeros((3, 1))

        # Converting individual arrays into a single array for
        # inital and target state
        x_init = np.vstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel))
        x_target = np.vstack((target_pos, target_rpy, target_vel, target_rpy_rates))    

        # HINTS: 
    # -----------------------------
    # - To add a constraint use
    #   constraints += [<constraint>] 
    #   i.e., to specify x <= 0, we would use 'constraints += [x <= 0]'
    # - To add to the cost, you can simply use
    #   'cost += <value>'
    # - Use @ to multiply matrices and vectors (i.e., 'A@x' if A is a matrix and x is a vector)
    # - A useful function to know is cp.quad_form(x, M) which implements x^T M x (do not use it for scalar x!)
    # - Use x[:, k] to retrieve x_k
  
        # For each stage in k = 0, ..., N-1
        for k in range(N):
            # Cost
            e = x[:, k] - x_target
            cost += cp.quad_form(e, weight_tracking) + cp.quad_form(u[:, k], weight_input)

            
            # constrains
            constraints += [x[:, k+1] == vehicle.A @ x[:, k] + vehicle.B @ u[:,k]]

            #print(cost)
    
    
    
        # EXERCISE: Implement the cost components and/or constraints that need to be added once, here
        # constrains
        constraints += [x[:, 0] == x_init]


        return rpm
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
    
    
	def _QuatToRot(self, quat):
		#normalize quat
		quat = quat / np.sqrt(sum(quat**2))
		
		q_hat[0,1] = -q[3]
		q_hat[0,2] = q[2]
		q_hat[1,2] = -q[1]
		q_hat[1,0] = q[3]
		q_hat[2,0] = -q[2]
		q_hat[2,1] = q[1]
		
		return np.eye(3) + 2*q_hat*q_hat + 2*q[0]*q_hat
	
