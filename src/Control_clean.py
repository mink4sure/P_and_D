import numpy as np
import pybullet
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
import casadi as cs

class MPC(BaseControl):
    def __init__(self, drone_model: DroneModel, g: float=9.8):

        super().__init__(drone_model=drone_model, g=g)

        # Getting parameters for drone control
        self.horizon = 10

        self.g = g
        self.m = self._getURDFParameter('m')
        self.l = self._getURDFParameter('arm')
        self.Ixx = self._getURDFParameter('ixx')
        self.Iyy = self._getURDFParameter('iyy')
        self.Izz = self._getURDFParameter('izz')

        # Calculating max producable force given max RPM
        self.RPM_max = 40000
        self.F_max = self.KF * self.RPM_max**2

        # Calculating max rate fo change of force given max dRPM
        self.dRPM_max = 1000
        self.dF_max = self.KF * 2 * self.dRPM_max
        
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

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                       cur_ang_vel, target_pos, target_rpy=np.zeros(3),
                        target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        
        dt = control_timestep
        
        # Converting current rotation Quat to een Rotations matrix
        cur_rpy = np.array(pybullet.getEulerFromQuaternion(cur_quat))
        
        # Rotating 
        cur_rpy = self.rotate45Z(cur_rpy)
        cur_ang_vel = self.rotate45Z(cur_ang_vel)
        target_rpy = self.rotate45Z(target_rpy)

        # Creating the optimizer object
        opti = cs.Opti()

        # Creating variables
        X = opti.variable(12, self.horizon+1)
        U = opti.variable(4, self.horizon)

        # Objective
        obj = 0
        for k in range(self.horizon):
            obj += 5 * (X[0:3, k] - target_pos).T @ (X[0:3, k] - target_pos)
            obj += (X[3:6, k] - target_vel).T @ (X[3:6, k] - target_vel)
            obj += (X[6:9, k] - target_rpy).T @ (X[6:9, k] - target_rpy)
            obj += (X[9:12, k] - target_rpy_rates).T @ (X[9:12, k] - target_rpy_rates)

        self.dynamics(optimizer=opti, X=X, U=U, dt=dt)

        # Constrains
        # intitial position etc
        opti.subject_to([X[0:3, 0] == cur_pos,
                        X[3:6, 0] == cur_vel,
                        X[6:9, 0] == cur_rpy,
                        X[9:, 0] == cur_ang_vel])

        # Getting the matrix to convert input U into F
        inv_f = cs.DM(4, 4)
        inv_f = np.linalg.inv(self.f)
        F = inv_f@U

        for k in range(self.horizon-1):
            # Setting limit on maximum Force
            opti.subject_to([
                F[:, k] >= np.zeros(4),
                F[:, k] <= np.ones(4) * self.F_max,
                (F[:, k]-F[:, k+1])**2 <= self.dF_max**2
                            ])

            # Setting limit on roll and pitch
            opti.subject_to([
                X[8, k] <= np.pi/180,
                X[8, k] >= -np.pi/180,
                X[9:11, k] <= 20/180 * np.pi
            ])






        # Letting the solver do it's work
        opti.solver('ipopt')
        opti.minimize(obj)
        sol = opti.solve()
        
        # Storing the input solution into F and converting it to RPM
        # not allowing any negative RPM
        F = inv_f @ sol.value(U)[:, 0]
        rpm_t = np.zeros(4)
        for i in range(4):
            if F[i] > 0:
                rpm_t[i] = (F[i]/self.KF)**(0.5)
            else:
                print('Negative force encountered :', F[i])
                rpm_t[i] = 0

        # Maybe switch around whith motor recieves what RPM (Not sure what the used definition is)
        lst = [0, 1, 2, 3]
        rpm = np.array([rpm_t[lst[0]], rpm_t[lst[1]], rpm_t[lst[2]], rpm_t[lst[3]]])

        #rpm = np.array([0, 0, 10000, 0])

        print('Found forces: ', F)
        print("Found RPM: ", rpm)

        return rpm

    def dynamics(self, optimizer, X, U, dt):
        # extracting state:
        p = X[0:3, :]
        v = X[3:6, :]
        o = X[6:9, :]
        w = X[9:, :]

        for k in range(self.horizon-1):
            # Giving the angles names for simplicity
            phi = o[0, k]
            theta = o[1, k]
            psi = o[2, k]

            # Calculating  second order derivatives based on the paper
            """ ddx = (u[0, k] * (casadi.sin(psi)*casadi.sin(phi) 
                    + casadi.cos(psi)*casadi.sin(theta) * casadi.cos(phi))) / self.m
            ddy = (u[0, k] * (casadi.sin(psi)*casadi.sin(theta)*casadi.cos(phi)
                    - casadi.cos(psi)*casadi.sin(phi))) / self.m
            ddz = (u[0, k]*(casadi.cos(psi)*casadi.cos(theta))) / self.m - self.g """
            ddx = cs.sin(theta) * U[0, k] / self.m
            ddy = cs.sin(phi) * U[0, k] / self.m
            ddz = U[0, k] /self.m - self.g

            ddphi = (U[2, k] * self.l) / self.Ixx
            ddtheta = (U[1, k] * self.l) / self.Iyy
            ddpsi = (U[3, k] * self.KM/self.KF) / self.Izz

            # Constrains for each step
            # Dynamics: In the global frame
            optimizer.subject_to([
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

    def rotate45Z(self, vect):
        # Getting rotation matrix to alling body frame with inertial axis
        rotZ = cs.DM(3, 3)
        rotZ = self.rotZ

        return rotZ@vect



