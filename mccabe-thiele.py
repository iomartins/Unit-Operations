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
        st_points = ([xW,xQ],[xW,yQ])
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
        # There are three points in each stage, namely P1, P2 and P3
        P1 = []
        P2 = []
        P3 = [(xD,xD)]
        
        points = []
        
        # Stage calculations loop
        while P3[-1][0] > xW:      # The procedure stops only when the minimum value,
            num += 1            # xW, is attained
            # Rectifying line as the base of the stage
            if P3[-1][0] >= xQ:
                # New points
                # Point 1, located on the rectifying line, equal to P3 calculated
                # at the previous step
                xP1 = P3[-1][0]
                yP1 = P3[-1][1]
                P1.append((xP1,yP1))
                
                # Point 2, located on the equilibrium curve, same vapor molar
                # fraction as Point 1
                # Equilibrium curve interpolator is used to calculate the liquid
                # molar fraction
                yP2 = yP1
                xP2 = self.eqcurve(yP2)
                P2.append((xP2,yP2))
                
                # Point 3, located on the operation line, the choice of the
                # operation line depends upon whether the liquid fraction is
                # higher than that calculated at the feed point; it has the
                # same liquid molar fraction as Point 2
                # Operation curve interpolator is used to calculate the vapor
                # molar fraction
                xP3 = xP2
                if xP3 >= xQ:
                    yP3 = self.rectline(xP3)
                else:
                    yP3 = self.stline(xP3)
                P3.append((xP3,yP3))
                
                # These variables store the coordinates in order to draw the
                # stages
                # x-coordinates
                x12 = (xP1,xP2)
                x23 = (xP2,xP3)
                
                # y-coordinates
                y12 = (yP1,yP2)
                y23 = (yP2,yP3)
                
                # Stage lines
                line_1 = [x12,y12]
                line_2 = [x23,y23]
                
                # List containing the coordinates of all stage lines
                points.append([line_1,line_2])
            # Stripping line as the base of the stage
            elif P3[-1][0] < xQ:
                # New points
                # Point 1, located on the rectifying line, equal to P3 calculated
                # at the previous step
                xP1 = P3[-1][0]
                yP1 = P3[-1][1]
                P1.append((xP1,yP1))
                
                # Point 2, located on the equilibrium curve, same vapor molar
                # fraction as Point 1
                # Equilibrium curve interpolator is used to calculate the liquid
                # molar fraction
                yP2 = yP1
                xP2 = self.eqcurve(yP2)
                P2.append((xP2,yP2))
                
                # Point 3, located on the stripping line, it has the same liquid
                # molar fraction as Point 2
                # Stripping curve interpolator is used to calculate the vapor
                # molar fraction
                xP3 = xP2
                try:
                    yP3 = self.stline(xP3)
                except:
                    yP3 = xP3
                P3.append((xP3,yP3))
                
                # These variables store the coordinates in order to draw the
                # stages
                # x-coordinates
                x12 = (xP1,xP2)
                x23 = (xP2,xP3)
                
                # y-coordinates
                y12 = (yP1,yP2)
                y23 = (yP2,yP3)
                
                # Stage lines
                line_1 = [x12,y12]
                line_2 = [x23,y23]
                
                # List containing the coordinates of all stage lines
                points.append([line_1,line_2])
        self.points = points
        self.number = num
                
            
    def theoretical_stages(self):
        
        xF, xD, xW = self.project_composition
        
        yA = np.arange(0,1.01,0.01)
        xA = self.eqcurve(yA)
        
        fig, axs = plt.subplots()
        axs.plot([0,1],[0,1],'k')
        axs.plot(xA,yA,'k')
        axs.plot(self.rect_points[0],self.rect_points[1],'#0080FF',label='Rectifying')
        axs.plot(self.feed_points[0],self.feed_points[1],'#FF8000',label='Feed')
        axs.plot(self.st_points[0],self.st_points[1],'#808080',label='Stripping')
        num = 0
        for line in self.points:
            num += 1
            axs.plot(line[0][0],line[0][1],color='lightsalmon',linestyle='--')
            axs.plot(line[1][0],line[1][1],color='lightsalmon',linestyle='--')
            axs.plot(line[0][0][1],line[0][1][1],linestyle=None,marker='o',
                     fillstyle='none',color='lightsalmon')
            axs.text(line[0][0][1]-0.03,line[0][1][1],str(num))
        axs.plot(0,0,color='lightsalmon',linestyle='--',label='Theoretical Stage')
        axs.plot([xD,xD],[0,xD],color='teal',linestyle='--')
        axs.text(0.95*xD,xD/2,'x$_D$')
        axs.plot([xF,xF],[0,xF],color='teal',linestyle='--')
        axs.text(1.05*xF,xF/2,'x$_F$')
        axs.plot([xW,xW],[0,xW],color='teal',linestyle='--')
        axs.text(1.05*xW,xW/2,'x$_W$')
        axs.grid()
        axs.legend()
        axs.set_xlabel('Liquid molar fraction of benzene [-]')
        axs.set_ylabel('Vapor molar fraction of benzene [-]')
        axs.set_xlim(0,1)
        axs.set_ylim(0,1)
            
        
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        