import math
import numpy as np
import pybullet
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

# For MPC control
import casadi

class MPC(BaseControl):
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
        self.m = self._getURDFParameter('m')

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
        cur_rpy = np.array(pybullet.getEulerFromQuaternion(cur_quat))

        # Setting horizon distance
        horizon = 20
        
        # Creating the optimizer object
        opti = casadi.Opti()

        # setting the variables
        p = opti.variable(3, horizon)   # Position
        v = opti.variable(3, horizon)   # Velocity
        o = opti.variable(3, horizon)   # Orientation: phi, theta, psi
        w = opti.variable(3, horizon)   # angulair velocity: d_phi, d_theta, d_psi
        u = opti.variable(4, horizon)   # Control input: w1, w2, w3, w4

        obj = 0

        
        for k in range(horizon-1):
            # Rotation matric from Drone to Global
            temp_R_a_to_b = np.array(pybullet.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)  # Hopefully rotation of drone written in Global frame basis
            R_a_to_b = casadi.DM(3,3)
            R_a_to_b = temp_R_a_to_b
            
            R_b_to_a = R_a_to_b.T

            phi = o[0, k]
            theta = o[1, k]
            psi = o[2, k]

            """ some_R = casadi.DX([[np.cos(theta), 0, -np.cos(psi)*np.sin(theta)],
                               [0, 1, np.sin(phi)],
                               [np.sin(theta), 0, np.cos(phi) * np.cos(theta)]])
 """
            #wb = some_R@w[:, k]

            # get force of rotors in global frame
            thrust_b = casadi.MX.zeros(3,1)
            for i in range(4):
                thrust_b[2, 0] += u[i, k] 
            
            thrust = R_b_to_a @ thrust_b

            # Cost for each step
            obj += (p[0, k] - target_pos[0])**2
            obj += (p[1, k] - target_pos[1])**2
            obj += (p[2, k] - target_pos[2])**2

            # Constrains for each step
            # Dynamics: In the global frame
            opti.subject_to([
                            p[:, k+1] == p[:, k] + dt * v[:, k],
                            v[:, k+1] == v[:, k] + dt * (np.array([0, 0, -self.g]) + thrust),
                            o[:, k+1] == o[:, k] + dt * w[:, k]

                            ])

        # Single time cost
        obj += 0

        # Single time contrains
        opti.subject_to([p[:, 0] == cur_pos,
                        v[:, 0] == cur_vel,
                        o[:, 0] == cur_rpy])
        

        opti.solver('ipopt')
        opti.minimize(obj)
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
	
