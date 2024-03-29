import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

import casadi as cs

class PIDMPCControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8, 
                 obstacles: list=[]
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

        # MPC variables
        self.horizon = 48
        self.T = 10
        self.Vmax = 0.5

        self.OBS = obstacles

        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
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
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

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

        traject_pos, traject_vel = self._MPCtraject(cur_pos=cur_pos, cur_vel=cur_vel, target_pos=target_pos, dt_simulation=control_timestep)
        print('current_pos: ', cur_pos)
        print('traject_pos: ', traject_pos)

        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         traject_pos,
                                                                         target_rpy,
                                                                         traject_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

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
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _MPCtraject(self, cur_pos, cur_vel, target_pos, dt_simulation):
        
        print("Target position: ", target_pos)
        #temp = p.getCollisionShapeData(self.OBS[0], -1, physicsClientId=env.CLIENT)
        #print(temp)

        obs1 = [0, 2, 1]

        dt = dt_simulation*self.T

        ### SOLVER OBJECT AND VARIABLES###
        opti = cs.Opti()
        X = opti.variable(6, self.horizon)


        pos_objs = [[2, -.5, .5], [2, .5, .5], [2, -1.5, .5], [2, 1.5, .5],
                    [4, 0, .5], [4, 0, 1.5], [4, 0, 3.5], [4, 0, 3.5]]


        ### COST ###
        q = 40
        obj = 0
        obj += (X[0:3, -1] - target_pos).T @ (X[0:3, -1] - target_pos)
        for k in range(self.horizon):
            #obj += (X[0:3, k] - target_pos).T @ (X[0:3, k] - target_pos)
            obj += ((X[0:3, k] - target_pos).T @ (X[0:3, k] - target_pos))
            #obj += 1/(X[2, k] )
       
            k1 = 1.2
            obj += k1 * q * self._costGaussian(size=[1, 1, 1], pos=[2, -.5, .5], X=X[:, k]) #eerste onder
            obj += k1 * q * self._costGaussian(size=[1, 1, 1], pos=[2, .5, .5], X=X[:, k]) #eerste onder
            obj += k1 * q * self._costGaussian(size=[1, 1, 1], pos=[2, -1.5, .5], X=X[:, k]) #eerste onder
            obj += k1 * q * self._costGaussian(size=[1, 1, 1], pos=[2, 1.5, .5], X=X[:, k]) #eerste onder

            # toren
            k2 = 1.5
            obj += k2 * q * self._costGaussian(size=[1, 1, 1], pos=[4, 0, .5], X=X[:, k]) #eerste onder
            obj += k2 * q * self._costGaussian(size=[1, 1, 1], pos=[4, 0, 1.5], X=X[:, k]) #eerste onder
            obj += k2 * q * self._costGaussian(size=[1, 1, 1], pos=[4, 0, 2.5], X=X[:, k]) #eerste onder
            obj += k2 * q * self._costGaussian(size=[1, 1, 1], pos=[4, 0, 3.5], X=X[:, k]) #eerste onder
            #obj += 12 * q * self._costGaussian(size=[1, 2, 2], pos=[4, 1, 1], X=X[:, k]) #eerste rechts boven
            #obj += q * self._costGaussian(size=[0.25, 2, 1], pos=[4, 0, 1.5],    X=X[:, k]) #tweede boven
            #obj += q * self._costGaussian(size=[0.25, 1, 1], pos=[4, 0.5, 0.5],  X=X[:, k]) #tweede links oder """

        
        ### Initial position constraint ###
        opti.subject_to([
            X[0:3, 0] == cur_pos,
            X[3:6, 0] == cur_vel
        ])

        ### Dynamics ###
        for k in range(self.horizon-1):
            opti.subject_to([
                X[0:3, k+1] == X[0:3, k] + dt * X[3:6, k],
           ])

        ### Contrains ###
        for k in range(1, self.horizon):
            opti.subject_to([
                X[3:6, k].T @ X[3:6, k] <= self.Vmax**2,
                #X[2, k] >= 0.1
                ])
            #opti.subject_to(cs.sqrt((X[0:3, k] - obs1).T @ (X[0:3, k] - obs1)) >= 1)

        s_opts = {'max_iterations': 5000}
        opti.solver('ipopt')
        opti.minimize(obj)
        sol = opti.solve()

        return sol.value(X)[0:3, 1], sol.value(X)[3:6, 1]


    def _costCube(self, X):
        r = (X[0])**2 + (X[1]-2)**2
        temp = -2000*(r)**2 + 100
        return cs.fmax(temp, 0)

    def _costCilinder(self, X):
        r_cyl = 1
        x_cyl = 0
        y_cyl = 2
        h_cyl = r_cyl**2 - (X[0]-x_cyl)**2 - (X[1]-y_cyl)**2
        return cs.fmax(h_cyl, 0)

    def _costGaussian(self, size, pos, X):
        beta = 2
        
        size_x = size[0]
        size_y = size[1]
        size_z = size[2]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        x = X[0]
        y = X[1]
        z = X[2]
        part_x = (x-pos_x)**beta/(2*size_x**2)
        part_y = (y-pos_y)**beta/(2*size_y**2)
        part_z = (z-pos_z)**beta/(2*size_z**2)

        norm_factor = 1/(size_x*size_y*size_z*2*np.pi*np.sqrt(2*np.pi))

        h_gaus = norm_factor * cs.exp(-(part_x + part_y + part_z))
        return h_gaus