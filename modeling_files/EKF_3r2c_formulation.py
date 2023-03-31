# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:23:09 2020

@author: rchintal
"""
import numpy as np
from numpy import dot
from filterpy.kalman import ExtendedKalmanFilter as EKF
import copy

class EKF_3r2c_formulation(EKF):
    def __init__(
            self, 
            dt_hr, 
            T_r, 
            T_w, 
            rc_params,
            df_data):
        EKF.__init__(self,8,1,3)
        self.T_r = T_r
        self.T_w = T_w
        self.y1 = rc_params["C_r_inv"]
        self.theta1 = rc_params["R_re_inv"]
        self.theta2 = rc_params["R_ra_inv"]
        self.y2 = rc_params["C_w_inv"]
        self.theta3 = rc_params["R_ea_inv"]
        self.dt = dt_hr
        self.alpha = rc_params["alpha"]
        self.EP_sim_data_pd = df_data
        self.pht = []
        self.inv_s = []
        
        self.x[0] = T_r
        self.x[1] = T_w
        self.x[2] = rc_params["C_r_inv"]
        self.x[3] = rc_params["R_re_inv"]
        self.x[4] = rc_params["R_ra_inv"]
        self.x[5] = rc_params["C_w_inv"]
        self.x[6] = rc_params["R_ea_inv"]
        self.x[7] = rc_params["alpha"]
        
    """ compute Jacobian of H matrix at x """
    def HJacobian_at(self,x):
        return np.array ([[1.,0.,0., 0, 0, 0,0,0]])
  
    """ compute measurement for slant range that
    would correspond to state x."""
    def hx(self,x):
        return x[0][0]
    
    def FJacobian_at(self,x,u):
        el11 = -x[2][0]*x[3][0] - x[2][0]*x[4][0]
        el12 = x[2][0]*x[3][0]
        el13 = x[3][0]*x[1][0] - x[3][0]*x[0][0]+ u[0]*x[4][0] -x[4][0]*x[0][0] +  x[7][0]*u[1] + u[2]
        el14 = x[2][0]*x[1][0] - x[2][0]*x[0][0]
        el15 = x[2][0]*u[0] - x[2][0]*x[0][0]
        el16 = 0
        el17 = 0
        el18 = x[2][0]*u[1]
        
        el21 = x[5][0]*x[3][0]
        el22 = -x[5][0]*x[3][0] - x[5][0]*x[6][0]
        el23 = 0
        el24 = x[5][0]*x[0][0] - x[5][0]*x[1][0]
        el25 = 0
        el26 = x[3][0]*x[0][0] - x[3][0]*x[1][0] + u[0]*x[6][0]-x[1][0]*x[6][0] + x[7][0]*u[1] + u[2]
        el27 = x[5][0]*u[0] - x[5][0]*x[1][0]
        el28 = x[5][0]*u[1]
    
        F = np.eye(8) + np.array([[el11, el12, el13, el14, el15, el16, el17, el18],
                                  [el21, el22, el23, el24, el25, el26, el27,el28],
                           [0, 0, 0, 0, 0, 0, 0,0],
                           [0, 0, 0, 0, 0, 0, 0,0],
                                 [0, 0, 0, 0, 0, 0, 0,0],
                                 [0, 0, 0, 0, 0, 0, 0,0],
                                 [0, 0, 0, 0, 0, 0, 0,0],
                                 [0, 0, 0, 0, 0, 0, 0,0]]) * self.dt
        return F
    
    def VJacobian_at(self,x,u):
        el11 = x[2][0]*x[4][0]
        el12 = x[2][0]*x[7][0]
        el13 = x[2][0]
    
        
        el21 = x[5][0]*x[6][0]
        el22 = x[5][0]*x[7][0] 
        el23 = x[5][0]
    
    
        V =  np.array([[el11, el12, el13],
            [el21, el22, el23],
           [0, 0, 0],
           [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
              [0, 0, 0]]) * self.dt
        return V
        
    def get_data(self,EP_sim_data_pd,k):
        """ Returns inputs and outputs at time step k
        """
        
        # EnergyPlus measured output
        T_measured = EP_sim_data_pd.T_room.iloc[k]
        self.T_r = T_measured
    
        # Energyplus inputs
        T_oa = EP_sim_data_pd.T_outdoor.iloc[k]
        Q_ghi = EP_sim_data_pd.Q_ghi.iloc[k]
        Q_load = EP_sim_data_pd.Q_load.iloc[k]
        Q_hvac = EP_sim_data_pd.Q_hvac.iloc[k]
        
        return np.array ([T_measured,T_oa,Q_ghi,Q_load, Q_hvac])
    
    def predict(self, u = 0):
        self.x = self.move(self.x,u,self.dt)
        self.F = self.FJacobian_at(self.x,u)
        M = np.array([[0, 0,0], 
                   [0, 0,0],
                  [0,0,1]])
        V = self.VJacobian_at(self.x,u)
        self.P = dot(self.F,self.P).dot(self.F.T) + dot(V, M).dot(V.T)
        #print('VmV^T is', dot(V,M).dot(V.T))
    
    def sim_predict(self,u, print_contributions = False):
        if not print_contributions:
            self.x = self.move(self.x,u,self.dt)
        else:
            self.x = self.move(self.x, u, self.dt, print_contributions = True)
        
    
    def move(self,x,u,dt, print_contributions = False):
        #print('x during move is', x)
        dTr = dt*x[2][0]*x[3][0]*(x[1][0] - x[0][0]) +\
              dt*x[2][0]*x[4][0]*(u[0] - x[0][0]) + \
              dt*x[2][0]*x[7][0]*u[1] + \
              dt*x[2][0]*u[2]
        dTw = dt*x[5][0]*x[3][0]*(x[0][0] - x[1][0]) + \
              dt*x[5][0]*x[6][0]*(u[0] - x[1][0]) + \
              dt*x[5][0]*x[7][0]*u[1] + \
              dt*x[5][0]*(u[2] - u[3]*0)
        dx = np.array([[dTr],[dTw],[0],[0],[0],[0],[0],[0]])
        return x + dx
    
    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(), n_pred = 1,k = 0,
               residual=np.subtract):
        """ Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, posterior is not computed

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, *args)
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.inv_s = np.linalg.inv(self.S)
        self.pht = PHT
        self.K = PHT.dot(np.linalg.inv(self.S))
        if self.K[0] < 0:
            self.P *= 0
            self.P[0,0] = 2.**2.
            self.P[1,1] = 2.**2.
            self.P[2,2] = 1000.
            self.P[3,3] = 1000.
            self.P[4,4] = 1000.
            self.P[5,5] = 1000.
            self.P[6,6] = 1000.
            self.P[7,7] = 0.2
            self.S = dot(H, PHT) + R
            self.inv_s = np.linalg.inv(self.S)
            self.pht = PHT
            self.K = PHT.dot(np.linalg.inv(self.S))
        hx = Hx(self.x, *hx_args)
        if n_pred > 1:
            self.x_pred = copy.deepcopy(self.x)
            self.res_pred = residual(z, hx)
            for i in range(n_pred -1):
                EP_data = self.get_data(self.EP_sim_data_pd,k + i)
                u = EP_data[1:]
                self.x_pred = self.move(self.x_pred,u,self.dt)
                if self.x_pred[0][0] > 12 and self.x_pred[0][0]< 35:
                    self.res_pred += abs(residual(self.EP_sim_data_pd.T_room[k+i+1], self.x_pred[0][0]))
                else:
                    self.res_pred += 15
        self.res = residual(z, hx)
        if n_pred > 1:
            self.x[0] += dot(self.K[0], self.res)
            self.x[1:] += dot(self.K[1:],self.res_pred)
        else:
            self.x = self.x + dot(self.K, self.res)
        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        #self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
