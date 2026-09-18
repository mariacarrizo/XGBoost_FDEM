""" Functions to forward model and perform individual and laterally constrained inversion"""

import empymod as ep
import numpy as np
from scipy.constants import mu_0
import pygimli as pg

def FDEM(sgm, thk, height=0.1, ):
    """ 1D FDEM response 
        
    Parameters
    ----------
    sgm : array
           Array of electrical conductivities [S/m], len(sigma) = nlay

    thk : array
           Array of thicknesses [m], len(thk) = nlay - 1

    Freq : frequency of device [Hz]

    coilOrient : array of strings
                  coil orientations: 'H' for horizontal coplanar, 
                  'V' for vertical coplanar, 'P' for perpendicular

    coilSpacing : array
                    separations of the transmitter and receiver coils [m]

    height : float
                height of the device with respect to ground [m]
                
    norm : normalize values with respect to offset

    Returns
    -------
    Array [OP, IP]

    """  
    # Set geometry
    Freq = 9000
    coilSpacing = [2, 4, 8]
    pcoilSpacing = [2.1, 4.1, 8.1]
    coilOrient = np.array(['H', 'V', 'P'])
    #height = 0

    # Source and receivers geometry [x, y, z]
    source    = [0, 0, -height]
    receivers = [coilSpacing, np.zeros(len(coilSpacing)), -height]    
    preceivers = [pcoilSpacing, np.zeros(len(pcoilSpacing)), -height]

    # Depth and resistivity
    res_air = 1e6
    res = np.hstack((res_air, 1/sgm))
    depth = np.hstack((0, np.cumsum(thk)))
    # Empty array to store responses
    HOP = []
    HIP = []
    POP = []
    PIP = []
    VOP = []
    VIP = []
    
    if any(coilOrient == 'H'):

        H_Hs = ep.dipole(source, receivers, depth, res, Freq, ab = 66, xdirect=None, 
                          verb=0)*(2j * np.pi * Freq * mu_0) 
        H_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime = Freq,
                         ab = 66, verb=0)*(2j * np.pi * Freq * mu_0)   
        op = (H_Hs/H_Hp).imag.amp()# * norm
        ip = (H_Hs/H_Hp).real.amp()# * norm
        HOP.append(op) 
        HIP.append(ip)

    if any(coilOrient == 'P'):
        # Maybe put 0.1m in receiver offset
        P_Hs = ep.dipole(source, preceivers, depth, res, Freq, ab=46, xdirect=None, 
                         verb=0)*(2j * np.pi * Freq * mu_0) 
        P_Hp = ep.dipole(source, preceivers, depth=[], res=[res_air], freqtime= Freq,
                         ab=66, verb=0)*(2j * np.pi * Freq * mu_0) 
        op = (P_Hs/P_Hp).imag.amp()# * norm
        ip = (P_Hs/P_Hp).real.amp()# * norm
        POP.append(op)
        PIP.append(ip)
        
    if any(coilOrient == 'V'):
        V_Hs = ep.dipole(source, receivers, depth, res, Freq, ab =55, xdirect=None, 
                         verb=0)*(2j * np.pi * Freq * mu_0) 
        V_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq, 
                         ab=55, verb=0)*(2j * np.pi * Freq * mu_0)
        op = (V_Hs/V_Hp).imag.amp()# * norm
        ip = (V_Hs/V_Hp).real.amp()# * norm
        VOP.append(op)
        VIP.append(ip)

    return np.array([HOP, HIP, VOP, VIP, POP, PIP]).ravel() 



def FDEM_field(sgm, thk, height=0.47,):
    """ 1D FDEM response 
        
    Parameters
    ----------
    sgm : array
           Array of electrical conductivities [S/m], len(sigma) = nlay

    thk : array
           Array of thicknesses [m], len(thk) = nlay - 1

    Freq : frequency of device [Hz]

    coilOrient : array of strings
                  coil orientations: 'H' for horizontal coplanar, 
                  'V' for vertical coplanar, 'P' for perpendicular

    coilSpacing : array
                    separations of the transmitter and receiver coils [m]

    height : float
                height of the device with respect to ground [m]
                
    norm : normalize values with respect to offset

    Returns
    -------
    Array [OP, IP]

    """  
    # Set geometry
    Freq = 9000
    coilSpacing = [2, 4, 8]
    pcoilSpacing = [2.1, 4.1, 8.1]
    coilOrient = np.array(['H', 'V'])
    #height = 0

    # Source and receivers geometry [x, y, z]
    source    = [0, 0, -height]
    receivers = [coilSpacing, np.zeros(len(coilSpacing)), -height]    
    preceivers = [pcoilSpacing, np.zeros(len(pcoilSpacing)), -height]

    # Depth and resistivity
    res_air = 1e6
    res = np.hstack((res_air, 1/sgm))
    depth = np.hstack((0, np.cumsum(thk)))
    # Empty array to store responses
    HOP = []
    HIP = []
    POP = []
    PIP = []
    VOP = []
    VIP = []
    
    if any(coilOrient == 'H'):

        H_Hs = ep.dipole(source, receivers, depth, res, Freq, ab = 66, xdirect=None, 
                          verb=0)*(2j * np.pi * Freq * mu_0) 
        H_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime = Freq,
                         ab = 66, verb=0)*(2j * np.pi * Freq * mu_0)   
        op = (H_Hs/H_Hp).imag.amp()# * norm
        ip = (H_Hs/H_Hp).real.amp() [1:]
        HOP.append(op) 
        HIP.append(ip)

    if any(coilOrient == 'P'):
        # Maybe put 0.1m in receiver offset
        P_Hs = ep.dipole(source, preceivers, depth, res, Freq, ab=46, xdirect=None, 
                         verb=0)*(2j * np.pi * Freq * mu_0) 
        P_Hp = ep.dipole(source, preceivers, depth=[], res=[res_air], freqtime= Freq,
                         ab=66, verb=0)*(2j * np.pi * Freq * mu_0) 
        op = (P_Hs/P_Hp).imag.amp()# * norm
        ip = (P_Hs/P_Hp).real.amp()# * norm
        POP.append(op)
        PIP.append(ip)
        
    if any(coilOrient == 'V'):
        V_Hs = ep.dipole(source, receivers, depth, res, Freq, ab =55, xdirect=None, 
                         verb=0)*(2j * np.pi * Freq * mu_0) 
        V_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq, 
                         ab=55, verb=0)*(2j * np.pi * Freq * mu_0)
        op = (V_Hs/V_Hp).imag.amp()# * norm
        ip = (V_Hs/V_Hp).real.amp()[1:]
        VOP.append(op)
        VIP.append(ip)

    return np.hstack((HOP, HIP, VOP, VIP)).ravel()


class FDEM1DModelling(pg.frameworks.Modelling):
    
    def __init__(self, nlay=9, height=0.1):
        self.nlay = nlay
        self.height = height
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        #print('im using this one')
                   
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = FDEM(np.asarray(par)[self.nlay-1:self.nlay*2-1],   # sigma
                      np.asarray(par)[:self.nlay-1],                  # thickness
                      height=self.height)
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')


class FDEM1DModelling_field(pg.frameworks.Modelling):
    
    def __init__(self, nlay=9):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        #print('im using this one')
                   
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = FDEM_field(np.asarray(par)[self.nlay-1:self.nlay*2-1],   # sigma
                      np.asarray(par)[:self.nlay-1]                  # thickness
                      )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')

class LCModelling(pg.frameworks.LCModelling):
    """2D Laterally constrained (LC) modelling.

    2D Laterally constrained (LC) modelling based on BlockMatrices.
    """

    def __init__(self, fop, normalized = True, **kwargs):
        """Parameters: fop class ."""
        self.normalized = normalized
        if self.normalized == True:
            print('Inversion is normalized')
        super().__init__(fop,  **kwargs)

    def initJacobian(self, dataVals, nLay):
        """Initialize Jacobian matrix.

        Parameters
        ----------
        dataVals : ndarray | RMatrix | list
            Data values of size (nSounding x Data per sounding).
            All data per sounding need to be equal in length.
            If they don't fit into a matrix use list of sounding data.
        """

        nPar = 1
        
        nSoundings = len(dataVals)

        self.createParametrization(nSoundings, nLayers = nLay, nPar = nPar)

        if self._jac is not None:
            self._jac.clear()
        else:
            self._jac = pg.matrix.BlockMatrix()

        self.fops1D = []
        nData = 0

        for i in range(nSoundings):
            kwargs = {}
            for key, val in self._fopKwargs.items():
                if hasattr(val, '__iter__'):
                    if len(val) == nSoundings:
                        kwargs[key] = val[i]
                    else:
                        kwargs[key] = [val]
                else:
                    kwargs[key] = val

            f = None
            if issubclass(self._fopTemplate, pg.frameworks.Modelling):
                f = self._fopTemplate(**kwargs)
            else:
                f = type(self._fopTemplate)(self.verbose, **kwargs)

            f.setMultiThreadJacobian(self._parPerSounding)

            self._fops1D.append(f)

            nID = self._jac.addMatrix(f.jacobian())
            #print(nID)
            self._jac.addMatrixEntry(nID, nData, self._parPerSounding * i)
            #print(self._jac)
            nData += len(dataVals[i])

        self._jac.recalcMatrixSize()
        print("Jacobian size:", self._jac.rows(), self._jac.cols(), nData)
        self.setJacobian(self._jac)
        
    def createJacobian(self, par):
        """Create Jacobian matrix by creating individual Jacobians."""
        mods = np.asarray(par).reshape(self._nSoundings, self._parPerSounding)

        for i in range(self._nSoundings):
            self._fops1D[i].createJacobian(mods[i])
            
        #print('Jacobian shape:',np.shape(self._jac))
        #return self._jac
    
    def response(self, par):
        """Cut together forward responses of all soundings."""
        
        mods = np.asarray(par).reshape(self._nSoundings, self._parPerSounding)

        resp = pg.Vector(0)
        for i in range(self._nSoundings):
            r = self._fops1D[i].response(mods[i])
            resp = pg.cat(resp, r)
        return resp
    
    def constraint_matrix(self, dataVals, nLay=None):
        """ Create constraint matrix

        Parameters
        ----------
            dataVals : list of 1D pyGIMLi data vectors (nSounding x Data per sounding) [list]
            nLayers : number of layers (for blocky inversion) [int]
        """

        nSoundings = len(dataVals) # number of soundings
        
        boundaries_thk = (nLay - 1) * (nSoundings - 1) # inner mesh boundaries for thicknesses
        boundaries_sig = nLay * (nSoundings - 1) # inner mesh boundaries for conductivities

        CM = np.zeros((boundaries_thk + boundaries_sig, (nLay *2 - 1) * nSoundings))
        h = -np.eye(1, nLay * 2) + np.eye(1, nLay * 2, k = (nLay * 2 - 1))

        for i in range(boundaries_thk + boundaries_sig):
            CM[i, i:h.shape[1]+i] = h

        #print('Size of constraint matrix:', np.shape(CM))

        self.CM = pg.utils.toSparseMatrix(CM) # convert to sparse pg matrix

        return self.CM
    
    def createWeight(self, dataVals, cWeight_1, cWeight_2, nLay=None):
        """ Create constraint weights (cWeights)
            Blocky model : vertical constraint weights for both model parameter regions

        Parameters
        ----------
            dataVals : list of 1D pyGIMLi data vectors [list]
            cWeight_1 : vertical constraint weight (smooth model) or thickness constraint weight
                        (blocky model) [float]
            cWeight_2 : horizontal constraint weight (smooth model) or resistivity constraint weight
                        (blocky model) [float]
            nLay : number of layers (for blocky inversion) [int]
        """
        nSoundings = len(dataVals) # number of soundings
        
        """ constraint weights for blocky model """
        cWeight_thk = cWeight_1
        cWeight_sig = cWeight_2

        boundaries_thk = (nLay - 1) * (nSoundings - 1)
        boundaries_sig = nLay * (nSoundings - 1)

        cWeight_thk = pg.Vector(boundaries_thk, cWeight_thk)

        cWeight_sig = pg.Vector(boundaries_sig, cWeight_sig)
        
        if self.normalized == True:
            self.cWeight = pg.cat(cWeight_thk, cWeight_sig) / self.norm * (boundaries_thk + boundaries_sig)
        else:
            self.cWeight = pg.cat(cWeight_thk, cWeight_sig)
        print('Constraint cWeight length:', len(self.cWeight))
        #return self.cWeight

    def createConstraints(self):
        """ create weighted constraint matrix """
        self._CW = pg.matrix.LMultRMatrix(self.CM, self.cWeight, verbose = True)
        self.setConstraints(self._CW)
        print('CW: ', self._CW)
        
    def normalization(self, dataVals, cWeight_1, cWeight_2, phiM_norm =[0, 0.5, 0.5, 1], nLay=None):
        """ Create normalization factor
        
        Parameters
        ----------
        dataVals : list of 1D pygimli data vectors
        cWeight_1 : thickness constraint weight
        cWeight_2 : electrical conductivity constraint weight
        phiM_norm : model objective function calculated by multiplying inversion 
                    starting model with constraint matrix for 4 cases
        nLay : number of layers
        """
        
        cWeight_thk = cWeight_1
        cWeight_sig = cWeight_2
        
        cWeight_ratio_thk = cWeight_thk / (cWeight_sig + cWeight_thk)
        cWeight_ratio_sig = cWeight_sig / (cWeight_sig + cWeight_thk)
        
        if cWeight_thk == cWeight_sig and cWeight_thk == 0:
            print('cWeights are zero')
            slope = phiM_norm[0]
            self.norm = 0
            raise ValueError('At least one cWeight should be non-zero')
            
        elif cWeight_thk == cWeight_sig and cWeight_thk != 0:
            print('cWeights are equal')
            slope = phiM_norm[3]
            self.norm = slope * cWeight_thk
            
        elif cWeight_sig == 0 and cWeight_thk != 0:
            print('only cWeight for thickness')
            slope = phiM_norm[2]
            self.norm = slope * cWeight_thk
            
        elif cWeight_thk == 0 and cWeight_sig != 0:
            print('only cWeight for resistivity')
            slope = phiM_norm[1]
            self.norm = slope * cWeight_sig
            
        elif cWeight_thk < cWeight_sig:
            print('cWeight for resistivity is dominant')
            slope_sig = phiM_norm[1]
            slope_thk = phiM_norm[2]
            self.norm = slope_sig * cWeight_sig + slope_thk * cWeight_thk * cWeight_ratio_thk
            
        else:
            print('cWeight for thickness is dominant')
            slope_sig = phiM_norm[1]
            slope_thk = phiM_norm[2]
            self.norm = slope_thk * cWeight_thk + slope_sig * cWeight_sig * cWeight_ratio_sig   

    def drawModel(self, ax, model, **kwargs):
        """Draw models as stitched 1D model section."""
        mods = np.asarray(model).reshape(self._nSoundings,
                                         self._parPerSounding)
        pg.viewer.mpl.showStitchedModels(mods, ax=ax, useMesh=True,
                                         x=self.soundingPos,
                                         **kwargs)