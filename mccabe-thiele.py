# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:41:41 2023

@author: iomartins
"""

import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

class McCabe_Thiele:
    
    def __init__(self,fluid_mixture,p):
        
        self.pressure = p
        self.mixture = fluid_mixture
    
    
    def set_volatilty(self,alpha_AB):
        
        xA = np.arange(0,1.1,0.01)
        yA = alpha_AB*xA/(1 + (alpha_AB - 1)*xA)
        self.eqcurve = interp1d(yA,xA)
    
    
    def set_externalfile(self,path):
        
        df = pd.read_csv(path)
        xA = df['xA']
        yA = df['yA']
        self.eqcurve = interp1d(yA,xA)
    
    
    def set_coolprop(self,mixture):
        
        return
    
    
    def inlet_configuration(self,F,T,comp):
        
        self.inlet_temperature = T
        self.inlet_stream = F
        self.project_composition = comp
        
        xF, xD, xW = comp
        D = ((xF - xW)/(xD - xW))*F
        W = F - D
        
        self.outlet_stream = [D,W]
        
    
    def operation_lines(self,R):
        
        self.reflux_ratio = R
        
        xF, xD, xW = self.project_composition
        D, W = self.outlet_stream
        F = self.inlet_stream
        
        # Defining the mixture string input to me be used in CoolProp
        mixture_composition = (
            self.mixture[0] + '[' + str(xF) + ']&' + self.mixture[1] +
            '[' + str(1 - xF) + ']')
        
        # Feed characteristics
        H_V = CP.PropsSI('HMOLAR','P',self.pressure,'Q',1,mixture_composition)
        H_L = CP.PropsSI('HMOLAR','P',self.pressure,'Q',0,mixture_composition)
        H_f = CP.PropsSI('HMOLAR','P',self.pressure,'T',self.inlet_temperature,
                         mixture_composition)
        q = (H_V - H_f)/(H_V - H_L)
        # Defining the x coordinate where feed and rectification lines meet
        if q >= 0.99 and q <= 1.01:
            xQ = 1
            yQ = 1
        else:
            xQ = (xF/(1 - q) - xD/(R + 1))/(R/(R + 1) - q/(q - 1))
            yQ = q*xQ/(q - 1) + xF/(1 - q)
        
        self.feed_quantity = q
        
        # Defining two points for the feed line
        feed_points = ([xF,xQ],[xF,yQ])
        self.feed_points = feed_points
        self.qline = interp1d(feed_points[0],feed_points[1])
        
        # Defining two points for the stripping line
        st_points = ([0,xQ],[-W/(R*D + q*F - W),yQ])
        self.st_points = ([xW,xQ],[xW,yQ])
        self.stline = interp1d(st_points[0],st_points[1])
        self.rect_points = ([xQ,xD],[yQ,xD])
        
        # Defining two points for the rectification line
        rect_points = ([0,xD],[xD/(R + 1),xD])
        self.rectline = interp1d(rect_points[0],rect_points[1])
        self.rect_points = ([xQ,xD],[yQ,xD])
        
        
    def lewis_sorel(self):
        
        xF, xD, xW = self.project_composition
        # Liquid molar fraction at the point where all lines meet
        xQ = self.feed_points[0][1]
        # Initializing variables for the stage calculations
        # Number of theoretical stages
        num = 0
        # x-coordinate of the equilibrium curve
        xE = [xD]
        # y-coordinate of the operation line
        xO = [xD]
        points = []
        
        # Stage calculations loop
        while xO[-1] > xW:      # The procedure stops only when the minimum value,
            num += 1            # xW, is attained
            # Rectifying line as the base of the stage
            if xE[-1] >= xQ:
                # Defining new coordinates
                xE.append(self.eqcurve(xO[-1]))
                xO.append(self.rectline(xE[-2]))
                
                # New points
                P1 = (xE[-2],xO[-1])
                P2 = (self.eqcurve(P1[1]),P1[1])
                P3 = (P2[0],self.rectline(xE[-1]))
                
                # x-coordinates
                x12 = (P1[0],P2[0])
                x23 = (P2[0],P3[0])
                
                # y-coordinates
                y12 = (P1[1],P2[1])
                y23 = (P2[1],P3[1])
                
                # Stage lines
                line_1 = [x12,y12]
                line_2 = [x23,y23]
                
                # List containing the coordinates of all stage lines
                points.append([line_1,line_2])
                print(xO[-1],'rect')
            # Stripping line as the base of the stage
            elif xE[-1] < xQ:
                # Defining new coordinates
                xE.append(self.eqcurve(xO[-1]))
                try:
                    xO.append(self.stline(xE[-2]))
                except:
                    xO.append(xW)
                
                # New points
                P1 = (xE[-2],xO[-1])
                print(P1)
                P2 = (self.eqcurve(P1[1]),P1[1])
                try:
                    P3 = (P2[0],self.stline(xE[-1]))
                except:
                    P3 = (P2[0],xW)
                
                # x-coordinates
                x12 = (P1[0],P2[0])
                x23 = (P2[0],P3[0])
                
                # y-coordinates
                y12 = (P1[1],P2[1])
                y23 = (P2[1],P3[1])
                
                # Stage lines
                line_1 = [x12,y12]
                line_2 = [x23,y23]
                
                # List containing the coordinates of all stage lines
                points.append([line_1,line_2])
                print(xO[-1],'strip')
        self.points = points
        self.number = num
                
            
    def theoretical_stages(self):
        
        xF, xD, xW = self.project_composition
        
        yA = np.arange(0,1.01,0.01)
        xA = self.eqcurve(yA)
        
        fig, axs = plt.subplots()
        axs.plot([0,1],[0,1],'k')
        axs.plot(xA,yA,'k')
        axs.plot(self.rect_points[0],self.rect_points[1],'k')
        axs.plot(self.feed_points[0],self.feed_points[1],'k')
        axs.plot(self.st_points[0],self.st_points[1],'k')
        for line in self.points:
            axs.plot(line[0][0],line[0][1],'k')
            axs.plot(line[1][0],line[1][1],'k')
        axs.plot([xD,xD],[0,xD],'k--')
        axs.plot([xF,xF],[0,xF],'k--')
        axs.plot([xW,xW],[0,xW],'k--')
        axs.grid()
        axs.set_xlabel('Liquid molar fraction of benzene [-]')
        axs.set_ylabel('Vapor molar fraction of benzene [-]')
            
        
        
p = 101325
fluid_mixture = ('benzene','toluene')
F = 100
T = 327.6
composition = [0.45,0.95,0.1]
R = 4

tower_1 = McCabe_Thiele(fluid_mixture,p)
tower_1.pressure
tower_1.set_externalfile('C:\\Users\\iomartins\\ex26.4.1.csv')
tower_1.inlet_configuration(F,T,composition)
tower_1.outlet_stream
tower_1.operation_lines(R)
tower_1.lewis_sorel()
tower_1.theoretical_stages()        
        
        
        
        
        
        
        
        
        
        
        
        
        