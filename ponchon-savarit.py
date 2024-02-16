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
        self._eqcurve = interp1d(xA,yA)
    
    
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
        self._eqcurve = interp1d(xA,yA)

    
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
        
        F = F/3.6               # Convertendo de kmol/h para mol/s
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
            H_V.append(CP.PropsSI('HMOLAR','P',self.pressure,'Q',1,mixture_composition))
            H_L.append(CP.PropsSI('HMOLAR','P',self.pressure,'Q',0,mixture_composition))
        
        self.vapor_enthalpy = interp1d(x1,H_V)
        self.liquid_enthalpy = interp1d(x1,H_L)
        
        # Vapor and liquid enthalpies at the first stage can be calculated since
        # the composition is known from the input
        # Calculation can be made from the interpolators or using the PropsSI
        # function from CoolProp
        mixture_composition = (
            self.mixture[0] + '[' + str(xD) + ']&' + self.mixture[1] +
            '[' + str(1 - xD) + ']')
        H1 = CP.PropsSI('HMOLAR','P',self.pressure,'Q',0,mixture_composition)
        hD = CP.PropsSI('HMOLAR','P',self.pressure,'Q',1,mixture_composition)
        Hlv = H1 - hD
        qCD = (R + 1)*Hlv
        
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
        # qrw parameter from Gomide (1988)
        self.qRW = qRW
        
        self.condenser = Qc
        self.reboiler = Qr
        
        # Inlet and outlet points
        _F = (xF,hF)
        _D = (xD,hD - qCD)
        _W = (xW,hW - qRW)
        _R = (xD,hD)
        
        self.project_points = (_F,_D,_W,_R)
        
        
    def theoretical_lines(self):
        
        xF, xD, xW = self.project_composition
        _F, _D, _W, _R = self.project_points
        hD = self.liquid_enthalpy(xD)
        hW = self.liquid_enthalpy(xW)
        
        # Creating interpolator for the first line: D' ---> R'
        _x = (_D[0],_R[0])
        _y = (_D[1],_R[1])
        dist_line = interp1d(_x,_y)
        # For each stage a new operation line interpolator will be created
        OL = [dist_line]
        yV = []
        xL = [xD]
        Hv = [self.vapor_enthalpy(xD)]
        Hl = [self.liquid_enthalpy(xD)]
        num = 0
        rect = 0
        stp = 0
        
        while xL[-1] > xW:
            num += 1
            # Rectifying section
            if xL[-1] >= xF:
                rect += 1
                # Finding the point where the operation line intersects the
                # saturated vapor line
                OL.append(interp1d([xL[-1],xD],[Hl[-1],hD + self.qCD],
                          fill_value='extrapolate'))
                def func1(x): 
                    return(OL[-1](x) - self.vapor_enthalpy(x))
                y = fsolve(func1,0.5)[0]
                yV.append(y)
                _hV = self.vapor_enthalpy(yV[-1])
                Hv.append(_hV)
                if num == 1: 
                    xL.pop(0)    # Cleaning the vector in the first loop
                # Finding the point in the saturated liquid enthalpy line with
                # the equilibrium curve
                x = self.eqcurve(y)
                xL.append(x)
                _hL = self.liquid_enthalpy(xL[-1])
                Hl.append(_hL)
            # Stripping section
            elif xL[-1] < xF:
                stp += 1
                # Finding the point where the operation line intersects the
                # saturated liquid line
                OL.append(interp1d([xW,xL[-1]],[hW + self.qRW,Hl[-1]],
                                   fill_value='extrapolate'))
                def func2(x):
                    return(OL[-1](x) - self.vapor_enthalpy(x))
                y = fsolve(func2,[0.5])[0]
                yV.append(y)
                _hV = self.vapor_enthalpy(yV[-1])
                Hv.append(_hV)
                if num == 1: 
                    xL.pop(0)    # Cleaning the vector in the first loop
                # Finding the point in the saturated liquid enthalpy line with
                # the equilibrium curve
                x = self.eqcurve(y)
                xL.append(x)
                _hL = self.liquid_enthalpy(xL[-1])
                Hl.append(_hL)
        self.number = num
        self.sections = (rect,stp)
        self.enthalpies = (Hv,Hl)
        self.compositions = (yV,xL)
                
    def draw_stages(self):
        
        xF, xD, xW = self.project_composition
        _F, _D, _W, _R = self.project_points
        hD = self.liquid_enthalpy(xD)
        hW = self.liquid_enthalpy(xW)
        comp = (
            self.mixture[0] + '[' + str(xF) + ']&' + self.mixture[1] +
            '[' + str(1 - xF) + ']')
        hF = CP.PropsSI('HMOLAR','P',self.pressure,'T',T,comp)
        Hv, Hl = self.enthalpies
        yV, xL = self.compositions
        
        x = np.arange(0,1.01,0.01)
                
        fig, axs = plt.subplots()
        axs.plot(x,self.vapor_enthalpy(x),'k')
        axs.plot(x[::10],self.vapor_enthalpy(x)[::10],'k',marker='v',
                 fillstyle='none',label='Vapor enthalpy')
        axs.plot(x,self.liquid_enthalpy(x),'k')
        axs.plot(x[::10],self.liquid_enthalpy(x)[::10],'k',marker='^',
                 fillstyle='none',label='Liquid enthalpy')
        axs.plot([xD,xD],[0,hD - self.qCD],color='teal',linestyle='--')
        axs.plot(xD,hD - self.qCD,linestyle='none',marker='o',color='#030764')
        axs.text(xD + 0.02,hD - self.qCD,'D')
        axs.plot(xD,hD,linestyle='none',marker='o',color='#030764')
        axs.text(xD + 0.02,hD,'R')
        axs.plot([xW,xW],[0,hW - self.qRW],color='teal',linestyle='--')
        axs.plot(xW,hW - self.qRW,linestyle='none',marker='o',color='#030764')
        axs.text(xW + 0.02,hW - self.qRW,'W')
        axs.plot(xF,hF,linestyle='none',marker='o',color='#030764')
        axs.text(xF + 0.02,hF,'F')
        axs.plot([0],[0],color='lightsalmon',linestyle='--',label='Operation line')
        axs.plot([0],[0],color='#DDA0DD',linestyle='-.',label='Equilibrium line')
        axs.plot([xW,xD],[hW - self.qRW,hD - self.qCD],linestyle=':')
        for j in range(self.number):
            if j <= self.sections[0]:
                axs.plot([xD,xL[j]],[hD - self.qCD,Hl[j]],color='lightsalmon',
                         linestyle='--')
                axs.plot([yV[j],xL[j+1]],[Hv[j],Hl[j+1]],color='#DDA0DD',
                         linestyle='-.')
            else:
                axs.plot([xW,yV[j]],[hW - self.qRW,Hv[j]],color='lightsalmon',
                         linestyle='--')
                axs.plot([yV[j-1],xL[j]],[Hv[j-1],Hl[j]],color='#DDA0DD',
                         linestyle='-.')
        axs.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        axs.set_xlim(0,1)
        axs.grid()
        axs.legend()
        axs.set_xlabel('Molar fraction of ' + self.mixture[0] + '[-]')
        axs.set_ylabel('Molar enthalpy [J mol$^{-1}$]')
                

# Data from Example 26.4-1 from Geankoplis et al. (2018)
p = 101325
fluid_mixture = ('benzene','toluene')
F = 100
T = 327.6
composition = [0.45,0.95,0.1]
R = 1.5

tower_1 = Ponchon_Savarit(fluid_mixture,p)
tower_1.set_coolprop()
tower_1.tower_configuration(F,R,composition,T)
tower_1.theoretical_lines()
tower_1.draw_stages()