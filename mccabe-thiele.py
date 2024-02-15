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
from scipy.optimize import fsolve
from math import log10

class McCabe_Thiele:
    """
    
    Class used to simulate an ideal distillation column using graphical method
    of McCabe-Thiele to represent the result.
    
    Attributes
    -----------
    
    mixture : list
        list with 2 elements, the components in the binary mixture
    
    pressure : float
        column pressure, Pa
    
    eqcurve : interp1d
        interpolator used to calculate the vapor molar fraction given a liquid
        molar fraction value
    
    feed_quality : float
        feed quality number, dimensionless
    
    inlet_stream : float
        feed molar flow, kmol/h
    
    project_composition : list
        list containing all project compositions regarding the most volatile
        component, [xF,xD,xW]
    
    inlet_temperature : float
        temperature of the feed stream, K
        
    outlet_stream : list
        list containing the distillate and waste molar flows, [D,W], kmol/h
    
    reflux_ratio : float
        reflux ratio value, dimensionless
        
    feed_points : list
        list containing the points of the feed line
    
    qline : interp1d
        interpolator used in the feed line
        
    st_points : list
        list containing the points of the stripping line
    
    stline : interp1d
        interpolator used in the stripping line
        
    rect_points : list
        list containing the points of the rectifying line
    
    rectline : interp1d
        interpolator used in the rectifying line
            
    points : list
        list containing the coordinates of all points of all lines in the theoretical
        stages
    
    number : int
        number of theoretical stages
        
    minimum_stages: float
        minimum number of stages calculated with Fenske equation
    
    minimum_reflux : float
        minimum reflux ratio calculated with Underwood equation
        
    Gilliland_stages : float
        number of stages calculated with Gilliland correlation
    
    vapor_composition : list
        list containing the vapor molar fraction leaving each stage
        
    liquid_flow : list
        list containing the liquid molar flow in each stage
        
    vapor_flow : list
        list containing the vapor molar flow in each stage
        
    stages_temperature : list
        list containing the temperature in each stage
    
    condenser : float
        energy consumed by the condenser, MW
        
    reboiler : float
        energy released by the reboiler, MW
    
    Methods
    -------
    
    set_volatility(alpha_AB)
        uses the relative volatility to create an interpolator to calculate the
        equilibrium vapor molar fraction given the liquid molar fraction
    
    set_externalfile(alpha)
        uses an external file to create an interpolator to calculate the 
        equilibrium vapor molar fraction given the liquid molar fraction
        
    set_coolprop(mixture)
        uses CoolProp to create an interpolator to calculate the equilibrium
        vapor molar fraction given the liquid molar fraction
        
    phase_diagram()
        uses CoolProp to plot the boiling and dew temperature as function of
        molar fraction
        
    set_feedquality(q)
        sets a custom value for the feed quality number
    
    inlet_configuration(F,comp,T)
        given the inlet molar (mass) flow, inlet temperature and project composition,
        the method calculates both outlet molar (mass) flows
        
    operation_lines(R)
        given the reflux ratio, this method creates an interpolator for each
        operation line, rectifying, stripping and waste
        
    lewis_sorel()
        uses the lewis-sorel method to calculate the number of theoretical stages
        and the molar fraction in each stage
        
    FUG_correlations()
        employs the Fenske, Underwood and Gilliland correlations to calculate
        values for Rmin, Nmin and N
    
    theoretical_stages()
        uses the points calculated in the lewis_sorel method and plots the figure
    
    molar_fraction()
        plots a figure with the vapor molar fraction per stage
    
    molar_flow()
        plots a figure with the vapor molar flow per stage
        
    heat_exchangers_demand(Hvap)
        calculates the energy transferred at the condenser and reboiler
        
    """
    
    
    def __init__(self,fluid_mixture,p=101325):
        """
        Parameters
        ----------
        
        fluid_mixture : list
            list with 2 elements, the components in the binary mixture
            
        p : float, optional
            pressure inside the column. The default pressure is 101325 Pa
        
        Returns
        -------
        None.

        """
    
        self.mixture = fluid_mixture
        self.pressure = p
        
        # Errors and exception messages regarding the inlet mixture input
        
        if type(self.mixture) != list and type(self.mixture) != tuple:
            raise TypeError('The mixture input must be a list or a tuple')
        
        if len(self.mixture) != 2:
            raise ValueError('There must be two components in the mixture input')
        
        if self.mixture[0] == self.mixture[1]:
            raise ValueError('There must be two different fluids')
        
        fluid_list = CP.get_global_param_string("FluidsList").split(',')
        if fluid_mixture[0].capitalize() not in fluid_list or fluid_mixture[1].capitalize() not in fluid_list:
            raise Warning('One or more fluids are absent in the CoolProp database')
            raise Warning('Double check if they were correctly written')
    
    
    def set_volatilty(self,alpha_AB):
        """
        Method to create an interpolator to determine the liquid molar fraction
        in equilibrium with a given vapor molar fraction. By using this method,
        the equilibrium curve is calculated considering an average relative
        volatility.
        
        Parameters
        ----------
        
        alpha_AB : float
            relative volatility between both components

        Returns
        -------
        None.
        
        """
        
        xA = np.arange(0,1.1,0.01)
        yA = alpha_AB*xA/(1 + (alpha_AB - 1)*xA)
        
        self.eqcurve = interp1d(yA,xA)
    
    
    def set_externalfile(self,path):
        """
        Method to create an interpolator to determine the liquid molar fraction
        in equilibrium with a given vapor molar fraction. By using this method,
        the equilibrium curve is built with information from an external .csv
        file.
        
        The .csv file must have the following format
        xA|yA|pA*|pB*|T
          |  |   |   |
          |  |   |   |
          |  |   |   |
          |  |   |   |
        
        Parameters
        ----------
        
        path : string
            path to a csv file containing liquid and molar fractions of
            the most volatile component
        
        Returns
        -------
        None.
        
        """
        
        df = pd.read_csv(path)
        xA = df['xA']
        yA = df['yA']
        
        self.eqcurve = interp1d(yA,xA)
    
    
    def set_coolprop(self):
        """
        Method to create an interpolator to determine the liquid molar fraction
        in equilibrium with a given vapor molar fraction. This method requires
        no input as it will consider the strings informed upon the object construction.
        Note that this method requires a valid input for the CoolProp package
        and the software must be able to calculate the dew and bubble temperatures
        of the informed mixture.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        
        """
        xA = np.arange(0,1.01,0.01)
        Tl = []
        Tv = []
        p = self.pressure
        
        try:
            for j in xA:
                mixture_composition = (
                    self.mixture[0] + '[' + str(j) + ']&' + self.mixture[1] +
                    '[' + str(1 - j) + ']')
                Tl.append(CP.PropsSI('T','P',p,'Q',0,mixture_composition))
                Tv.append(CP.PropsSI('T','P',p,'Q',1,mixture_composition))
            
            Tl_curve = interp1d(xA,Tl)
            Tv_curve = interp1d(Tv,xA)
            yA = []
            
            for x in xA:
                T = Tl_curve(x)
                yA.append(Tv_curve(T))
            
            self.eqcurve = interp1d(yA,xA)
            self._eqcurve = interp1d(xA,yA)
        
        except:
            raise ValueError('CoolProp was unable to perform bubble and dew temperature calculations, choose another input method')
        
    
    def phase_diagram(self):
        """
        Method to plot the phase diagram using CoolProp.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        x1 = np.arange(0,1.05,0.05)
        Tl = []
        Tv = []
        p = self.pressure
        
        try:
            for j in x1:
                mixture_composition = (
                    self.mixture[0] + '[' + str(j) + ']&' + self.mixture[1] +
                    '[' + str(1 - j) + ']')
                Tl.append(CP.PropsSI('T','P',p,'Q',0,mixture_composition))
                Tv.append(CP.PropsSI('T','P',p,'Q',1,mixture_composition))
                
            fig, axs = plt.subplots()
            axs.plot(x1,Tl,'k-.',label='Saturated liquid')
            axs.plot(x1,Tv,'k--',label='Saturated vapor')
            axs.fill_between(x1,Tl,Tv,color='#C875C4',label='VLE Region')
            axs.grid()
            axs.legend()
            axs.set_xlabel('Molar fraction of ' + self.mixture[0])
            axs.set_ylabel('Temperature [K]')
            axs.set_xlim(0,1)
        
        except:
            raise ValueError('CoolProp was unable to perform bubble and dew temperature calculations, choose another input method')

    
    def set_feedquality(self,q):
        """
        Method to set the quality number of the feed stream. This method must be
        used when CoolProp cannot calculate the molar enthalpy at the feed and
        saturated liquid and vapor conditions.
        
        Parameters
        ----------
        
        q : float
            feed quality parameter

        Returns
        -------
        None.
        
        """
        
        self.feed_quality = q


    def inlet_configuration(self,F,comp,T=300):
        """
        Method to set the quality number of the feed stream. This method must be
        used when CoolProp cannot calculate the molar enthalpy at the feed and
        saturated liquid and vapor conditions.
        
        Parameters
        ----------
        
        F : float
            inlet molar flow, must be given in kmol/h
            
        comp : list
            list containing the project compositions, [xF,xD,xW] or the feed,
            distillate and waste molar fraction, respectively. These values
            refer to the most volatile component
        
        T : float, optional
            temperature of the feed stream, if a feed stream quality value is
            informed, no temperature value is required

        Returns
        -------
        None.
        
        """
                    
        self.inlet_stream = F
        self.project_composition = comp
        self.inlet_temperature = T
        try:
            xF, xD, xW = comp
        except:
            raise TypeError('The project composition must have the feed, distillate and waste molar fractions')
        
        D = ((xF - xW)/(xD - xW))*F
        W = F - D
        
        self.outlet_stream = [D,W]
        
    
    def operation_lines(self,R):
        """
        Method to create the interpolators for the operation lines: rectifying,
        feed and stripping. Each interpolator requires two points, since it is
        a straight line and the second point in all three interpolator is the
        same, referring to the feed stage. If you have any questions regarding
        the mass balances, refer to Chapter 26 of Geankoplis et al. (2018).
        
        Parameters
        ----------
        
        R : float
            reflux ratio, given by the ratio between liquid molar flow at the
            enrichment section and the distillate molar flow

        Returns
        -------
        None.
        
        """
        
        self.reflux_ratio = R
        
        xF, xD, xW = self.project_composition
        D, W = self.outlet_stream
        
        # Feed quality parameter (q) can be given as an input with the set_feedquality
        # method or calculated from the temperature using CoolProp
        try:
            q = self.feed_quality
        
        except:
            # Defining the mixture string input to me be used in CoolProp
            mixture_composition = (
                self.mixture[0] + '[' + str(xF) + ']&' + self.mixture[1] +
                '[' + str(1 - xF) + ']')
            
            # Feed characteristics
            try:
                H_V = CP.PropsSI('HMOLAR','P',self.pressure,'Q',1,mixture_composition)
                H_L = CP.PropsSI('HMOLAR','P',self.pressure,'Q',0,mixture_composition)
                H_f = CP.PropsSI('HMOLAR','P',self.pressure,'T',self.inlet_temperature,
                                 mixture_composition)
                q = (H_V - H_f)/(H_V - H_L)
                self.feed_quality = q
            
            except:
                raise ValueError('CoolProp could not calculate the enthalpy values for the given mixture')
        
        # Defining the x coordinate where feed and rectification lines meet
        if q >= 0.99 and q <= 1.01:    # Placeholder, it will be corrected
            xQ = 1                     # in later updates 
            yQ = 1
        else:
            xQ = (xF/(1 - q) - xD/(R + 1))/(R/(R + 1) - q/(q - 1))
            yQ = q*xQ/(q - 1) + xF/(1 - q)
                    
        # Defining two points for the feed line
        feed_points = ([xF,xQ],[xF,yQ])
        self.feed_points = feed_points
        self.qline = interp1d(feed_points[0],feed_points[1])
        
        # Defining two points for the stripping line
        st_points = ([xW,xQ],[xW,yQ])
        self.st_points = ([xW,xQ],[xW,yQ])
        self.stline = interp1d(st_points[0],st_points[1])
        self.rect_points = ([xQ,xD],[yQ,xD])
        
        # Defining two points for the rectification line
        rect_points = ([0,xD],[xD/(R + 1),xD])
        self.rectline = interp1d(rect_points[0],rect_points[1])
        self.rect_points = ([xQ,xD],[yQ,xD])
        
        
    def FUG_correlations(self):
        
        """
        Method containing the correlations of Fenske, Underwood and Gilliland to
        estimate theoretical values for minimal number of stages and reflux ratio
        and the number of stages for a given reflux ratio. For a thorough explanation
        regarding the correlations, refer to Chapter 8 of Wankat (2012) and
        Chapter 26 of Geankoplis et al. (2018).
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.
        
        """
        
        q = self.feed_quality
        xF, xD, xW = self.project_composition
        D, W = self.outlet_stream
        R = self.reflux_ratio
        
        # Determining the average volatility
        yT = self.vapor_composition[0]
        xT = self.eqcurve(yT)
        yB = self.vapor_composition[-1]
        xB = self.eqcurve(yB)
        
        alphaT = (yT/xT)*((1 - xT)/(1 - yT))
        alphaB = (yB/xB)*((1 - xB)/(1 - yB))
        alpha = (alphaT*alphaB)**(1/2)
        
        # Fenske equation
        ratio = (xD/(1 - xD))*((1 - xW)/xW)
        Nmin = log10(ratio)/log10(alpha)
        self.minimum_stages = Nmin
        
        # Determining curve interpolator for an extended q-line
        a = q/(q - 1)
        b = -xF/(q - 1)
        # y = a*x + b
        x1 = 0
        y1 = a*x1 + b
        x2 = 1
        y2 = a*x2 + b
        self._qline = interp1d([0,1],[y1,y2])
        
        def pinch_point(x):
            
            res = self._eqcurve(x) - self._qline(x)
            
            return(res)
        
        xP = fsolve(pinch_point,0.5)[0]
        yP = self._qline(xP)
        
        # Underwood equation
        ratio = (xD - yP)/(xD - xP)
        Rmin = ratio/(1 - ratio)
        self.minimum_reflux = Rmin
        
        
        # Gilliand (1940) correlation, adapted by Liddle (1968)
        X = (R - Rmin)/(R + 1)
        
        if X >= 0 and X <= 0.01:
            Y = 1 - 18.5715*X
        
        elif X > 0.01 and X < 0.9:
            Y = 0.545827 - 0.591422*X + 0.002743/X
            
        elif X >= 0.9 and X <= 1:
            Y = 0.16595 - 0.16595*X
        
        else:
            raise ValueError('Values outside the range for Gilliland correlation')
        
        N = (Y + Nmin)/(1 - Y)
        self.Gilliland_stages = N
        
        print('Minimum number of stages: ' + str(Nmin))
        print('Minimum reflux ratio: ' + str(Rmin))
        print('Number of stages with Gilliland correlation: ' + str(N))
        
        
    def lewis_sorel(self):
        """
        Method to apply the Lewis-Sorel method to calculate the number of theoretical
        steps in an ideal binary distillation. The present formulation considers
        a stage to be formed by three distinct points, namely P1, P2 and P3.
        
        P2__________P1     P2 is located at the equilibrium line while P1 and P3
         |                 P3 are located at the operation line. P1 and P2 have
         |                 the same y coordinate; P2 and P3 have the same x coordinate.
         |                 P1 of a given stage has the P3 coordinates of the last
         |                 stage.
         |
        P3

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        
        """
        
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
            num += 1               # xW, is attained
            if num == 100:
                raise ValueError('Convergence was not obtained')
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
                    self.feed_stage = num
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
        """
        Method to plot the theoretical stages on an equilibrium curve along the
        operation lines. This method must be executed after the lewis_sorel
        method.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        
        xF, xD, xW = self.project_composition
        
        yA = np.arange(0,1.01,0.01)
        xA = self.eqcurve(yA)
        try:
            fig, axs = plt.subplots()
            axs.plot([0,1],[0,1],'k')
            axs.plot(xA,yA,'k')
            axs.plot(self.rect_points[0],self.rect_points[1],'#0080FF',
                     label='Rectifying line')
            axs.plot(self.feed_points[0],self.feed_points[1],'#FF8000',
                     label='Feed line')
            axs.plot(self.st_points[0],self.st_points[1],'#808080',
                     label='Stripping line')
            num = 0
            for line in self.points:
                num += 1
                axs.plot(line[0][0],line[0][1],color='lightsalmon',linestyle='--')
                axs.plot(line[1][0],line[1][1],color='lightsalmon',linestyle='--')
                axs.plot(line[0][0][1],line[0][1][1],linestyle=None,marker='o',
                         fillstyle='none',color='lightsalmon')
                axs.text(line[0][0][1]-0.03,line[0][1][1],str(num))
            axs.plot(0,0,color='lightsalmon',linestyle='--',label='Theoretical stage')
            axs.plot([xD,xD],[0,xD],color='teal',linestyle='--')
            axs.text(0.95*xD,xD/2,'x$_D$')
            axs.plot([xF,xF],[0,xF],color='teal',linestyle='--')
            axs.text(1.05*xF,xF/2,'x$_F$')
            axs.plot([xW,xW],[0,xW],color='teal',linestyle='--')
            axs.text(1.05*xW,xW/2,'x$_W$')
            axs.grid()
            axs.legend()
            axs.set_xlabel('Liquid molar fraction of ' + str(self.mixture[0]) + ' [-]')
            axs.set_ylabel('Vapor molar fraction of ' + str(self.mixture[0]) + ' [-]')
            axs.set_xlim(0,1)
            axs.set_ylim(0,1)
        
        except:
            raise ValueError('The lewis_sorel method must be executed before the present one')
        
        
    def molar_fraction(self):
        """
        Method to calculate the vapor molar fraction leaving each stage. This 
        method must be executed after the lewis_sorel method.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        xF, xD, xW = self.project_composition
        stages = np.arange(1,self.number+1,1)
        
        lines = []
        for line in self.points:
            lines.append(line[0][1][1])
        self.vapor_composition = lines
        
        fig, axs = plt.subplots()
        axs.plot(stages,lines,linestyle='-.',marker='v',fillstyle='none',color='k')
        axs.plot(1,xD,linestyle='--',marker='^',fillstyle='none',
                 color='teal',label='Distillate composition')
        axs.plot([1,self.feed_stage],[xF,xF],color='teal',linestyle='--')
        axs.plot(self.feed_stage,xF,linestyle='--',marker='o',fillstyle='none',
                 color='teal',label='Feed composition')
        axs.plot([1,self.number],[xW,xW],color='teal',linestyle='--')
        axs.plot(self.number,xW,linestyle='--',marker='s',fillstyle='none',
                 color='teal',label='Waste composition')
        axs.grid()
        axs.legend()
        axs.set_xlabel('Theoretical stage')
        axs.set_ylabel('Molar fraction of ' + str(self.mixture[0]) + ' [-]')
        axs.set_xlim(1,self.number+0.2)
        axs.set_ylim(0,1)
        
        
    def molar_flow(self):
        """
        Method to calculate the liquid and vapor molar flow leaving each stage.
        This method must be executed after the operation_lines method, as the mass
        balance depend upon the reflux ratio value.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        xF, xD, xW = self.project_composition
        D, W = self.outlet_stream
        F = self.inlet_stream
        q = self.feed_quality
        R = self.reflux_ratio
        
        # Rectifying section
        L_r = R*D
        V_r = L_r + D
        
        # Stripping section
        L_s = L_r + q*F
        V_s = V_r + (1 - q)*F
        
        L = []
        V = []
        stages = np.arange(1,self.number+1,1)
        for j in stages:
            if j <= self.feed_stage:
                L.append(L_r)
                V.append(V_r)
            else:
                L.append(L_s)
                V.append(V_s)
        
        self.liquid_flow = L
        self.vapor_flow = V
        
        # Drawing the graph

        fig, axs = plt.subplots()
        axs.plot(stages,L,linestyle='-.',marker='o',fillstyle='none',color='teal',
                 label='Liquid')
        axs.plot(stages,V,linestyle='-.',marker='s',fillstyle='none',
                 color='lightcoral',label='Vapor')
        axs.grid()
        axs.legend()
        axs.set_xlabel('Theoretical stage')
        axs.set_ylabel('Molar flow [kmol h$^{-1}$]')
        
    
    def temperature(self):
        """
        Method to calculate the temperature in each stage using the saturation
        pressure and solving the non-linear system with the fsolve function from
        scyPy. The saturation pressure behavior as function from temperature was
        modeled using CoolProp. This method must be executed after the lewis_sorel 
        method.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        def psat1(T):
            psat1 = CP.PropsSI('P','Q',1,'T',T,self.mixture[0])
            return(psat1)
        
        p = self.pressure
        stages = np.arange(1,self.number+1,1)
        y = self.vapor_composition
        x = self.eqcurve(y)
        
        T = []
        try:
            for j,stage in enumerate(stages,0):
                x1 = x[j]
                y1 = y[j]
                def f(T):
                    res = y1*p - x1*psat1(T)
                    return(res)
                
                T0 = 273.15 + 50
                T.append(fsolve(f,T0))
        # A methodology to calculate the saturation pressure from an external file
        # will be implemented in the future
        except: 
            raise ValueError('Saturation pressure could not be calculated with CoolProp')
        
        self.stages_temperature = T
        fig, axs = plt.subplots()
        axs.plot(stages,T,color='k',linestyle='-.',marker='o',fillstyle='none')
        axs.grid()
        axs.set_xlabel('Theoretical stage')
        axs.set_ylabel('Temperature [K]')
    
        
    def heat_exchangers_demand(self,Hvap=1):
        """
        Method to estimate the energy demand at the condenser and reboiler. The
        vaporization molar enthalpy in each heat exchanger can be provided as a
        list, [Hvap_cond,Hvap_reb], or they can be calculated from CoolProp. 
        
        Parameters
        ----------
        
        Hvap : list, optional
            list containing the molar vaporization enthalpy, J/mol, of the 
            condenser and reboiler.

        Returns
        -------
        
        condenser : float
            energy consumed at the condenser, MW
        
        reboiler : float
            energy released at the reboiler, MW

        """
        
        L = self.liquid_flow[-1]
        V = self.vapor_flow[0]
        
        if Hvap != 1:
            condenser = V*(-Hvap[0])/3600
            reboiler = L*Hvap[1]/3600
            
        else:
            T_top = self.stages_temperature[0]
            T_bottom = self.stages_temperature[-1]
            y = self.vapor_composition[0]
            x = self.eqcurve(self.vapor_composition[-1])
            try:
                # Condenser
                comp = (self.mixture[0] + '[' + str(y) + ']&' +
                        self.mixture[1] + '[' + str(1 - y) + ']')
                H_V = CP.PropsSI('HMOLAR','T',T_bottom,'Q',1,comp)
                H_L = CP.PropsSI('HMOLAR','T',T_top,'Q',0,comp)
                Hvap = H_V - H_L
                condenser = V*(-Hvap)/3600
                
                # Reboiler
                comp = (self.mixture[0] + '[' + str(x) + ']&' +
                        self.mixture[1] + '[' + str(1 - x) + ']')
                H_V = CP.PropsSI('HMOLAR','T',T_bottom,'Q',1,comp)
                H_L = CP.PropsSI('HMOLAR','T',T_top,'Q',0,comp)
                Hvap = H_V - H_L
                reboiler = L*Hvap/3600
            
            except:
                raise ValueError('Molar enthalpy could not be calculated with CoolProp')
            
            print('Heat exchanged at condenser: ' + str(condenser/1e3) + ' MW')
            print('Heat exchanged at reboiler: ' + str(reboiler/1e3) + ' MW')
        
        self.condenser = condenser/1e3
        self.reboiler = reboiler/1e3
        
        return(condenser,reboiler)


# Data from Example 26.4-1 from Geankoplis et al. (2018)
p = 101325
fluid_mixture = ('benzene','toluene')
F = 100
T = 327.6
composition = [0.45,0.95,0.1]
R = 4

tower_1 = McCabe_Thiele(fluid_mixture,p)
tower_1.set_coolprop()
tower_1.inlet_configuration(F,composition,T)
tower_1.operation_lines(R)
tower_1.lewis_sorel()
tower_1.theoretical_stages()
tower_1.molar_fraction()
tower_1.molar_flow()
tower_1.temperature()
tower_1.FUG_correlations()
tower_1.heat_exchangers_demand()