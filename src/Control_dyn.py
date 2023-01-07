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

        # Getting parameters for drone control
        self.g = g
        self.m = self._getURDFParameter('m')
        self.l = self._getURDFParameter('arm')
        self.Ixx = self._getURDFParameter('ixx')
        self.Iyy = self._getURDFParameter('iyy')
        self.Izz = self._getURDFParameter('izz')
        
        # f is defined as U = f @ F so the following
        # interpretations can be made in combination with
        # the used dynamical equations
        self.f = np.array([[1, 1, 1, 1],    # Total thust of the system
                            [-1, 0, 1, 0],  # Forces that cause a rotation theta (around y-axis)
                            [0, 1, 0, -1],  # Forces that cause a rotation phi (around x-axis)
                            [1, -1, 1, -1]])# Relationship between produced forces and resulting
                                            # moments about the z-axis (related to angle psi)

        self.rotZ = np.sqrt(2)/2 * np.array([[1, -1, 0],
                                             [1, 1, 0],
                                             [0, 0, 2/np.sqrt(2)]])

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

        print("target pos: ", target_pos)

        # Setting horizon distance
        horizon = 5
        dt = control_timestep
        
        # Creating the optimizer object
        opti = casadi.Opti()

        # setting the variables
        p = opti.variable(3, horizon)   # Position
        v = opti.variable(3, horizon)   # Velocity
        o = opti.variable(3, horizon)   # Orientation: phi, theta, psi
        w = opti.variable(3, horizon)   # angulair velocity: d_phi, d_theta, d_psi
        u = opti.variable(4, horizon)   # Control input: f * F

        obj = 0

        # Calculating max producable force given max RPM
        RPM_max = 40000
        omega_max = RPM_max * 2 * np.pi
        F_max = self.KF * RPM_max**2

        # Calculating max rate fo change of force given max dRPM
        dRPM_max = 1000
        dF_max = self.KF * 2 * dRPM_max

        # Getting the matrix to convert input U into F
        inv_f = casadi.DM(4, 4)
        inv_f = np.linalg.inv(self.f)

        # Converting current rotation Quat to een Rotations matrix
        cur_rpy = np.array(pybullet.getEulerFromQuaternion(cur_quat))
        
        # Getting rotation matrix to alling body frame with inertial axis
        rotZ = casadi.DM(3, 3)
        rotZ = self.rotZ

        # Checking rotation matrix: Look's good!
        #print(self.rotZ)
        #print(rotZ)

        # Get orientations in bodyfrme allinged with axis of inertia
        cur_rpy = rotZ@cur_rpy
        target_rpy = self.rotZ@target_rpy

        #print('rpy thingies')
        #print(cur_rpy)
        #print(target_rpy)

        for k in range(horizon-1):
            # Giving the angles names for simplicity
            phi = o[0, k]
            theta = o[1, k]
            psi = o[2, k]
            

            ### COST ###
            # Cost for position tracking
            Kv = 1
            obj += Kv * (p[0, k] - target_pos[0])**2
            obj += Kv * (p[1, k] - target_pos[1])**2
            obj += Kv * (p[2, k] - target_pos[2])**2
            
            #print('Check on position error X: ', (p[0, k] - target_pos[0])**2)
            #print('Check on position error Z: ', (p[2, k] - target_pos[2])**2)
            #print(target_pos[0])

            # Cost for keeping Yaw == 0
            #obj += casadi.sin(psi)**2
            #obj += psi**2
            obj += casadi.cos(psi)

            
            
            ### CONTROL ###
            # Calculating  second order derivatives based on the paper
            ddx = (u[0, k] * (casadi.sin(psi)*casadi.sin(phi) 
                    + casadi.cos(psi)*casadi.sin(theta) * casadi.cos(phi))) / self.m
            ddy = (u[0, k] * (casadi.sin(psi)*casadi.sin(theta)*casadi.cos(phi)
                    - casadi.cos(psi)*casadi.sin(phi))) / self.m
            ddz = (u[0, k]*(casadi.cos(psi)*casadi.cos(theta))) / self.m - self.g

            #print("ddx: ", ddx)
            #print("ddy: ", ddy)
            #print("ddz: ", ddz)

            ddphi = (u[2, k] * self.l) / self.Ixx
            ddtheta = (u[1, k] * self.l) / self.Iyy
            ddpsi = (u[3, k] * self.KM/self.KF) / self.Izz

            # Constrains for each step
            # Dynamics: In the global frame
            opti.subject_to([
                            p[0, k+1] == p[0, k] + dt * v[0, k],
                            p[1, k+1] == p[1, k] + dt * v[1, k],
                            p[2, k+1] == p[2, k] + dt * v[2, k],
                            v[0, k+1] == v[0, k] + dt * ddx, 
                            v[1, k+1] == v[1, k] + dt * ddy, 
                            v[2, k+1] == v[2, k] + dt * ddz, 
                            o[0, k+1] == o[0, k] + dt * w[0, k],
                            o[1, k+1] == o[1, k] + dt * w[1, k],
                            o[2, k+1] == o[2, k] + dt * w[2, k],
                            w[0, k+1] == w[0, k] + dt * ddphi,
                            w[1, k+1] == w[1, k] + dt * ddtheta,
                            w[2, k+1] == w[2, k] + dt * ddpsi,
                            ])

            # Converting inputs to forces to set constrains
            temp_F_k = inv_f @ u[:, k]
            temp_F_kp1 = inv_f @ u[:, k+1]
            
            # Constrains
            dddeg = 5   # Constant for maximum rate of change
            deg = 5     # Constraint for 
            max_v = 1
            opti.subject_to([
                            #u[0, k] <= 4 * F_max,
                            #u[1, k] <= F_max,
                            #u[2, k] <= F_max,
                            #u[3, k] <= 2 * F_max,
                            # setting a max Speed
                            #v[0, k] <= max_v, 
                            #v[1, k] <= max_v,
                            #v[2, k] <= max_v,
                           
                            # setting min and max producable forces
                            temp_F_k[0] >= 0,
                            temp_F_k[1] >= 0, 
                            temp_F_k[2] >= 0,
                            temp_F_k[3] >= 0,
                            temp_F_k[0] <= F_max,
                            temp_F_k[1] <= F_max, 
                            temp_F_k[2] <= F_max, 
                            temp_F_k[3] <= F_max,

                            # setting a maximum rate fo chane for the actuators
                            (temp_F_k[0] - temp_F_kp1[0])**2 <= dF_max,
                            (temp_F_k[1] - temp_F_kp1[0])**2 <= dF_max, 
                            (temp_F_k[2] - temp_F_kp1[0])**2 <= dF_max, 
                            (temp_F_k[3] - temp_F_kp1[0])**2 <= dF_max,
                            
                            # setting a max rate of change for the anlges
                            #ddphi <= dddeg,
                            #ddphi >= -dddeg,
                            #ddtheta <= dddeg,
                            #ddtheta >= -dddeg,
                            #ddpsi <= dddeg,
                            #ddpsi >= -dddeg,
                            
                            # Setting maximum angles
                            #phi <= deg,
                            #phi >= -deg,
                            #theta <= deg,
                            #theta >= -deg,
                            #psi <= 0.5,
                            #psi >= -0.5
                            ])

        ############################################################

        # Single time cost
        #obj = (p[0, -1] - target_pos[0])**2 + (p[1, -1] - target_pos[1])**2 + (p[2, -1] - target_pos[2])**2 

        # Single time contrains
        opti.subject_to([p[:, 0] == cur_pos,
                        v[:, 0] == cur_vel,
                        o[:, 0] == cur_rpy,
                        w[:, 0] == cur_ang_vel])
        

        opti.solver('ipopt')
        opti.minimize(obj)
        sol = opti.solve()
        
        opti.debug.value(ddx)

        # Storing the input solution into F and converting it to RPM
        # not allowing any negative RPM
        F = inv_f @ sol.value(u)[:, 0]
        rpm_t = np.zeros(4)
        for i in range(4):
            if F[i] > 0:
                rpm_t[i] = (F[i]/self.KF)**(0.5)
            else:
                print('Negative force encountered :', F[i])
                rpm_t[i] = 0

        # Maybe switch around whith motor recieves what RPM (Not sure what the used definition is)
        lst = [0, 1, 2, 3]
        rpm = np.array([rpm_t[lst[0]], rpm_t[lst[1]], rpm_t[lst[2]], rpm_t[lst[3]]], dtype = int)

        #rpm = np.array([0, 0, 10000, 0])

        print('Found forces: ', F)
        print("Found RPM: ", rpm)

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
	
