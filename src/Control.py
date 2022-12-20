import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

# For MPC control
import casadi

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

        self.g = g

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

        '''
        opti = casadi.Opti()

        x = opti.variable()
        y = opti.variable()

        opti.minimize(  (y-x**2)**2   )
        opti.subject_to( x**2+y**2==1 )
        opti.subject_to(       x+y>=1 )

        opti.solver('ipopt')


        sol = opti.solve()

        print(sol.value(x))
        print(sol.value(y))
        '''

        dt = control_timestep

        # Converting current rotation Quat to een Rotations matrix
        cur_rpy = np.zeros((3))

        # Setting horizon distance
        horizon = 20
        
        # Creating the optimizer object
        opti = casadi.Opti()

        # setting the variables
        p = opti.variable(3, horizon)   # Position
        v = opti.variable(3, horizon)   # Velocity
        o = opti.variable(3, horizon)   # Orientation: phi, theta, psi
        w = opti.variable(1, horizon)   # Yaw rate
        u = opti.variable(4, horizon)   # Control input: phi_d, theta_d, yaw_rate_d, vertical velocity
        
        obj = 0

        
        for k in range(horizon-1):
            # Cost for each step
            obj += p[:, k] - target_pos

            # Constrains for each step
            # dynamics
            phi = o[0, k]
            theta = o[1, k]
            psi = o[2, k]

            T_z = 0.3367
            T_phi = 0.2386
            T_theta = 0.2386
            K_z = 1.227
            K_phi = 1.0181
            K_theta = 1.0167

            opti.subject_to([p[:, k+1] == p[:, k] + dt * v[:, k],   # position update
                            v[0, k+1] == v[0, k] + dt * self.g * np.tan(theta)/np.cos(phi),   # x Velocity
                            v[1, k+1] == v[1, k] + dt * self.g * np.tan(phi),    # y Velocity
                            v[2, k+1] == v[2, k] + dt * (K_z*u[3, k] - v[2, k])/T_z,    # z Velocity
                            o[0, k+1] == o[0, k] + dt * (K_phi*u[0, k] - phi)/T_phi,
                            o[1, k+1] == o[1, k] + dt * (K_theta*u[1, k] - theta)/T_theta,
                            o[2, k+1] == o[2, k] + dt * w[k],
                            w[k+1] == u[2, k] ])

        # Single time cost
        obj += 0

        # Single time contrains
        opti.subject_to([p[:, 0] == cur_pos,
                        v[:, 0] == cur_vel,
                        o[:, 0] == cur_rpy])
        
        opti.solver('ipopt')
        sol = opti.solve()
        
        print("Found controll values: ", sol.value(u)[:, 0])

        rpm = [0, 0, 0, 0]

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
        q_hat = np.zeros((3,3))
		

        q_hat[0,1] = -quat[3]
        q_hat[0,2] = quat[2]
        q_hat[1,2] = -quat[1]
        q_hat[1,0] = quat[3]
        q_hat[2,0] = -quat[2]
        q_hat[2,1] = quat[1]
        
        return np.eye(3) + 2*q_hat*q_hat + 2*quat[0]*q_hat
	
