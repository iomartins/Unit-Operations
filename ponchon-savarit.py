import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from math import log10

class Ponchon_Savarit:
    
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
            raise Exception('One or more fluids are absent in the CoolProp database')
            raise Exception('Double check if they were correctly written')
            
            
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
    
    
    def tower_configuration(self,F,R,comp,T=300):
        """
        Method to set the quality number of the feed stream. This method must be
        used when CoolProp cannot calculate the molar enthalpy at the feed and
        saturated liquid and vapor conditions.
        
        Parameters
        ----------
        
        F : float
            inlet molar flow, must be given in kmol/h
            
        R : float
            reflux ratio, given by the ratio between liquid molar flow at the
            enrichment section and the distillate molar flow
            
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
        self.reflux_ratio = R
        try:
            xF, xD, xW = comp
        except:
            raise TypeError('The project composition must have the feed, distillate and waste molar fractions')
        
        D = ((xF - xW)/(xD - xW))*F
        W = F - D
        self.outlet_stream = [D,W]
        
        # Calculating thermal loads at the condenser and reboiler
        # Creating interpolators for the upper and lower enthalpy lines
        x1 = np.arange(0,1.01,0.01)
        H_L = []
        H_V = []
        
        for j in x1:
            mixture_composition = (
                self.mixture[0] + '[' + str(j) + ']&' + self.mixture[1] +
                '[' + str(1 - j) + ']')
            H_V.append(CP.PropsSI('HMOLAR','P',self.pressure,'Q',0,mixture_composition))
            H_L.append(CP.PropsSI('HMOLAR','P',self.pressure,'Q',0,mixture_composition))
        
        self.vapor_enthalpy = interp1d(x1,H_V)
        self.liquid_enthalpy = interp1d(x1,H_L)
        
        # Vapor and liquid enthalpies at the first stage can be calculated since
        # the composition is known from the input
        # Calculation can be made from the interpolators or using the PropsSI
        # function from CoolProp
        H1 = self.vapor_enthalpy(xD)
        hD = self.liquid_enthalpy(xD)
        Hlv = H1 - hD
        qCD = - (R + 1)*Hlv
        
        # qcd parameter from Gomide (1988)
        self.qCD = qCD
        # Heat removed from the condenser
        Qc = qCD*D
        
        # Heat added to the reboiler, expression taken from the global 
        # energy balance of the distillation tower
        comp = (
            self.mixture[0] + '[' + str(xW) + ']&' + self.mixture[1] +
            '[' + str(1 - xW) + ']')
        hW = CP.PropsSI('HMOLAR','P',self.pressure,'Q',0,comp)
        comp = (
            self.mixture[0] + '[' + str(xF) + ']&' + self.mixture[1] +
            '[' + str(1 - xF) + ']')
        # The feed enthalpy must have a temperature input, since its physical
        # stage cannot be fixed
        hF = CP.PropsSI('HMOLAR','P',self.pressure,'T',T,comp)
        # Heat added to the reboiler
        Qr = D*hD + W*hW - F*hF - Qc
        qRW = Qr/W
        
        self.condenser = Qc
        self.reboiler = Qr
        
        
        
                
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        