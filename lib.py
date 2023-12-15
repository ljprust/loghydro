import numpy as np
import scipy.special as ss

def shocktube(rho, v, P, nCells, boxSize, x, gamma) :

    xDiscont   = 0.5   # initial position of discontinuity
    P1         = 1.0 # 100.0 # pressure on left
    P2         = 1.0   # pressure on right
    rho1       = 1.0 # 10.0  # density on left
    rho2       = 1.0   # density on right
    v1         = 0.0   # velocity on left
    v2         = 0.0   # velocity on right

    for i in range(0,nCells) :
        rho[i] = rho1
        v[i] = v1
        P[i] = P1
        if (x[i] > xDiscont) :
            rho[i] = rho2
            v[i]   = v2
            P[i]   = P2
    return rho, v, P

def gaussian(rho, v, P, nCells, boxSize, x, gamma) :

    x0 = 0.5
    v0 = 1.0
    P0 = 1.0/gamma
    C1 = 0.1
    C2 = 100.0

    for i in range(0,nCells) :
        rho[i] = 1.0 + C1*np.exp(-C2*(x[i]-x0)**2.0)
        v[i]   = v0
        P[i]   = P0

    return rho, v, P

def checkSolutionGaussian(rho, nCells, x, t) :

    x0 = 0.5
    v0 = 1.0
    C1 = 0.1
    C2 = 100.0

    rhoAnalytic = 1.0 + C1*np.exp(-C2*(x-x0-v0*t)**2.0)
    L1error = 1.0/float(nCells)*np.absolute(rho-rhoAnalytic).sum()
    print ' L1 error:',L1error

def linearWave(rho, v, P, nCells, boxSize, x, gamma) :

    rho0 = 1.0
    v0   = 0.0
    P0   = 1.0/gamma
    amplitude = 1.0e-3

    kx = 2.0*np.pi/boxSize
    for i in range(0,nCells) :
        rho[i] = rho0 * (1.0 + amplitude*np.sin(kx*x[i]))
        v[i]   = amplitude*np.cos(kx*x[i])
        P[i]   = P0*gamma*kx * (1.0 + amplitude*np.cos(kx*x[i]))

    return rho, v, P

def addEdges(array, begin, end) :
    con = np.concatenate( ([begin],array,[end]), axis=0 )
    return con

def minmod(x, y, z) :
    result = 0.25 * np.absolute( np.sign(x) + np.sign(y) ) \
    * ( np.sign(x) + np.sign(y) ) \
    * np.minimum( np.minimum( np.absolute(x), np.absolute(y) ), np.absolute(z)  )
    return result

def reconstruct(cm1, c0, cp1, cp2, theta) :
    cL = c0 + 0.5 * minmod( theta*(c0-cm1), 0.5*(cp1-cm1), theta*(cp1-c0) )
    cR = cp1 - 0.5 * minmod( theta*(cp1-c0), 0.5*(cp2-c0), theta*(cp2-cp1) )
    return cL, cR

def getCovarianceMatrix(x_stencil, delta, R, l) :
    denom = np.sqrt(2.0)*l/delta
    stencilSize = R+R+1
    C = np.zeros((stencilSize,stencilSize))
    deltakh = np.zeros((stencilSize,stencilSize))
    for k in range(0,stencilSize) :
        for h in range(0,stencilSize) :
            deltakh[k,h] = (x_stencil[k]-x_stencil[h])/delta
    plusfactor = (deltakh+1.0)/denom
    minusfactor = (deltakh-1.0)/denom
    C = np.sqrt(np.pi)*l*l/delta/delta*((plusfactor*ss.erf(plusfactor)+
    minusfactor*ss.erf(minusfactor))+1.0/np.sqrt(np.pi)*(np.exp(-plusfactor*plusfactor)
    +np.exp(-minusfactor*minusfactor))-2.0*(deltakh/denom*ss.erf(deltakh/denom)
    +1.0/np.sqrt(np.pi)*np.exp(-deltakh*deltakh/denom/denom)))
    return C

def getPredictionVector(x_stencil, xstar, delta, R, l) :
    denom = np.sqrt(2.0)*l/delta
    stencilSize = R+R+1
    TT = np.zeros(stencilSize)
    deltakstar = np.zeros(stencilSize)
    for k in range(0,stencilSize) :
        deltakstar[k] = (x_stencil[k]-xstar)/delta
    TT = np.sqrt(np.pi/2.0)*l/delta*(ss.erf((deltakstar+0.5)/denom)
    -ss.erf((deltakstar-0.5)/denom))
    return TT

def getWeightVector(C, TT) :
    Cinv = np.linalg.inv(C)
    zT = np.matmul(TT,Cinv)
    return zT

def reconstructGP(zT, U_stencil, gamma) :
    Grho, Gvel, Gpres = getState3(U_stencil, gamma)
    fstarbarrho  = np.matmul(zT,Grho)
    fstarbarvel  = np.matmul(zT,Gvel)
    fstarbarpres = np.matmul(zT,Gpres)
    return fstarbarrho, fstarbarvel, fstarbarpres

def RiemannGP(U, gamma, deltax, zTL, zTR, Rstencil) :
    Ucopy = np.copy(U) # need to stop python from retroactively changing stuff
    nCells = len(deltax)
    nFaces = nCells-1

    # extract cell-centered state variables
    rho, v, P = getState3(Ucopy, gamma)

    # initialize left and right states assuming 1st spatial order
    rhoL, rhoR = splitScalar(rho)
    vL, vR     = splitScalar(v)
    PL, PR     = splitScalar(P)

    for j in range(Rstencil, nFaces-Rstencil) :
        U_stencilL = Ucopy[:,j-Rstencil:j+Rstencil+1]
        U_stencilR = Ucopy[:,j-Rstencil+1:j+Rstencil+2]
        rhoL[j], vL[j], PL[j] = reconstructGP(zTR, U_stencilL, gamma)
        rhoR[j], vR[j], PR[j] = reconstructGP(zTL, U_stencilR, gamma)
        '''
        if (j==50) : # nFaces-Rstencil-1) :
            print ' rho L R:',rhoL[j],rhoR[j]
            print ' vel L R:',vL[j],vR[j]
            print ' P   L R:',PL[j],PR[j]
        '''
    # remake U and Fcent with reconstructed variables
    UL = buildU3(rhoL, vL, PL, gamma)
    UR = buildU3(rhoR, vR, PR, gamma)
    FcentL = buildFcent3(rhoL, vL, PL, gamma)
    FcentR = buildFcent3(rhoR, vR, PR, gamma)

    # get sound speed
    cL = getc(gamma, PL, rhoL)
    cR = getc(gamma, PR, rhoR)

    # find eigenvalues
    lambdaP_L = vL + cL
    lambdaP_R = vR + cR
    lambdaM_L = vL - cL
    lambdaM_R = vR - cR
    alphaP = max3(  lambdaP_L,  lambdaP_R )
    alphaM = max3( -lambdaM_L, -lambdaM_R )
    alphaMax = np.maximum( alphaP.max(), alphaM.max() )

    # find face fluxes
    Fface = np.zeros([3,nCells+1])
    Fface[:,1:nCells] = ( alphaP * FcentL + alphaM * FcentR - alphaP * alphaM * (UR - UL) ) / ( alphaP + alphaM )
    FfaceL, FfaceR = splitVector(Fface)

    # find time derivatives
    L = - ( FfaceR - FfaceL ) / deltax

    return L, alphaMax

def RiemannRecon(U, gamma, deltax, theta) :
    nCells = len(deltax)

    # extract state variables
    rho, v, P = getState3(U, gamma)

    # split state variables for reconstruction
    rhom1, rho0, rhop1, rhop2 = splitRecon(rho)
    vm1, v0, vp1, vp2 = splitRecon(v)
    Pm1, P0, Pp1, Pp2 = splitRecon(P)

    # do the reconstruction
    rhoL, rhoR = reconstruct(rhom1, rho0, rhop1, rhop2, theta)
    vL, vR = reconstruct(vm1, v0, vp1, vp2, theta)
    PL, PR = reconstruct(Pm1, P0, Pp1, Pp2, theta)

    # remake U and Fcent with reconstructed variables
    UL = buildU3(rhoL, vL, PL, gamma)
    UR = buildU3(rhoR, vR, PR, gamma)
    FcentL = buildFcent3(rhoL, vL, PL, gamma)
    FcentR = buildFcent3(rhoR, vR, PR, gamma)

    # get sound speed
    cL = getc(gamma, PL, rhoL)
    cR = getc(gamma, PR, rhoR)

    # find eigenvalues
    lambdaP_L = vL + cL
    lambdaP_R = vR + cR
    lambdaM_L = vL - cL
    lambdaM_R = vR - cR
    alphaP = max3( lambdaP_L, lambdaP_R )
    alphaM = max3( -lambdaM_L, -lambdaM_R )
    alphaMax = np.maximum( alphaP.max(), alphaM.max() )

    # find face fluxes
    Fface = np.zeros([3,nCells+1])
    Fface[:,2:nCells-1] = ( alphaP * FcentL + alphaM * FcentR - alphaP * alphaM * (UR - UL) ) / ( alphaP + alphaM )
    FfaceL, FfaceR = splitVector(Fface)

    # find time derivatives
    L = - ( FfaceR - FfaceL ) / deltax

    return L, alphaMax

def Riemann(U, gamma, deltax) :
    nCells = len(deltax)

    # extract state variables
    rho, v, P = getState3(U, gamma)

    # get cell-centered fluxes
    Fcent = buildFcent3(rho, v, P, gamma)

    # split into left and right values
    UL, UR = splitVector(U)
    FcentL, FcentR = splitVector(Fcent)

    # get sound speed
    c = getc(gamma, P, rho)

    # find eigenvalues
    lambdaP = v + c
    lambdaM = v - c
    lambdaP_L, lambdaP_R = splitScalar(lambdaP)
    lambdaM_L, lambdaM_R = splitScalar(lambdaM)
    alphaP = max3( lambdaP_L, lambdaP_R )
    alphaM = max3( -lambdaM_L, -lambdaM_R )
    alphaMax = np.maximum( alphaP.max(), alphaM.max() )

    # find fluxes at faces
    Fface = np.zeros([3,nCells+1])
    Fface[:,1:nCells] = ( alphaP * FcentL + alphaM * FcentR - alphaP * alphaM * (UR - UL) ) / ( alphaP + alphaM )
    FfaceL, FfaceR = splitVector(Fface)

    # find time derivatives
    L = - ( FfaceR - FfaceL ) / deltax

    return L, alphaMax

def resetGhosts(U, Nghosts, periodic=False) :
    length = U.shape[1]

    if (periodic) :
        for j in range(0,Nghosts) :
            for k in range(0,3) :
                U[k,j] = U[k,length-2*Nghosts+j]
                #print 'copying cell',length-2*Nghosts+j,'into cell',j
        for j in range(1,Nghosts+1) :
            for k in range(0,3) :
                U[k,length-j] = U[k,2*Nghosts-j]
                #print 'copying cell',2*Nghosts-j,'into cell',length-j

    else : # reflective
        for j in range(0,Nghosts) :
            for k in range(0,3) :
                U[k,j] = U[k,2*Nghosts-1-j]
        for j in range(1,Nghosts+1) :
            for k in range(0,3) :
                U[k,length-j] = U[k,length-2*Nghosts-1+j]

    return U

def resetMirror(U, doMirror, mirrorCell) :
    if doMirror :

        U[0,mirrorCell-1] =  U[0,mirrorCell-2]
        U[1,mirrorCell-1] = -U[1,mirrorCell-2]
        U[2,mirrorCell-1] =  U[2,mirrorCell-2]

        U[0,mirrorCell] =  U[0,mirrorCell-3]
        U[1,mirrorCell] = -U[1,mirrorCell-3]
        U[2,mirrorCell] =  U[2,mirrorCell-3]

        U[0,mirrorCell+1] =  U[0,mirrorCell+4]
        U[1,mirrorCell+1] = -U[1,mirrorCell+4]
        U[2,mirrorCell+1] =  U[2,mirrorCell+4]

        U[0,mirrorCell+2] =  U[0,mirrorCell+3]
        U[1,mirrorCell+2] = -U[1,mirrorCell+3]
        U[2,mirrorCell+2] =  U[2,mirrorCell+3]

    return U

def smoothMirror(U, doMirror, mirrorCellX, mirrorCellY) :
    if doMirror :
        U[:, mirrorCellX, mirrorCellY] = U[:, mirrorCellX-1, mirrorCellY]
    return U

def getE3(P, gamma, rho, v) :
    e = P / (gamma - 1.) / rho
    E = rho * ( e + 0.5 * v * v )
    return E

def getE4(P, gamma, rho, vx, vy) :
    e = P / (gamma - 1.) / rho
    E = rho * ( e + 0.5 * ( vx*vx + vy*vy ) )
    return E

def getc(gamma, P, rho) :
    c = np.sqrt( gamma * P / rho )
    return c

def max3(array1, array2) :
    arrayMax = np.maximum( 0., np.maximum( array1, array2 ) )
    return arrayMax

def buildU3(rho, v, P, gamma) :
    length = len(rho)
    U = np.zeros([3,length])
    E = getE3(P, gamma, rho, v)
    U[0,0:length] = rho
    U[1,0:length] = rho * v
    U[2,0:length] = E
    return U

def buildU4(rho, vx, vy, P, gamma) :
    shape = rho.shape
    nx = shape[0]
    ny = shape[1]
    U = np.zeros([4,nx,ny])
    E = getE4(P, gamma, rho, vx, vy)
    U[0,:,:] = rho
    U[1,:,:] = rho * vx
    U[2,:,:] = rho * vy
    U[3,:,:] = E
    return U

def buildUArray(rho, vx, vy, P, gamma) :
    shape = rho.shape
    nx = shape[0]
    U = np.zeros([4,nx])
    E = getE4(P, gamma, rho, vx, vy)
    U[0,:] = rho
    U[1,:] = rho * vx
    U[2,:] = rho * vy
    U[3,:] = E
    return U

def getState3(U, gamma) :
    rho = U[0,:]
    v = U[1,:] / rho
    e = U[2,:] / rho - 0.5 * v * v
    P = ( gamma - 1.0 ) * rho * e
    return rho, v, P

def getState4(U, gamma) :
    rho = U[0,:,:]
    vx = U[1,:,:] / rho
    vy = U[2,:,:] / rho
    e = U[3,:,:] / rho - 0.5 * ( vx*vx + vy*vy )
    P = ( gamma - 1.0 ) * rho * e
    return rho, vx, vy, P

def getStateArray(U, gamma) :
    rho = U[0,:]
    vx = U[1,:] / rho
    vy = U[2,:] / rho
    e = U[3,:] / rho - 0.5 * ( vx*vx + vy*vy )
    P = ( gamma - 1.0 ) * rho * e
    return rho, vx, vy, P

def getCons3(U, nCells) :
    cons1 = U[0,1:(nCells-1)].sum()
    cons2 = U[1,1:(nCells-1)].sum()
    cons3 = U[2,1:(nCells-1)].sum()
    return cons1, cons2, cons3

def getCons4(U) :
    cons1 = U[0,:,:].sum()
    cons2 = U[1,:,:].sum()
    cons3 = U[2,:,:].sum()
    cons4 = U[3,:,:].sum()
    return cons1, cons2, cons3, cons4

def buildFcent3(rho, v, P, gamma) :
    length = len(rho)
    Fcent = np.zeros([3,length])
    E = getE3(P, gamma, rho, v)
    Fcent[0,0:length] = rho * v
    Fcent[1,0:length] = rho * v * v + P
    Fcent[2,0:length] = ( E + P ) * v
    return Fcent

def buildFcent4(rho, vx, vy, P, gamma) :
    shape = rho.shape
    nx = shape[0]
    ny = shape[1]
    Fcent = np.zeros([4,nx,ny])
    E = getE4(P, gamma, rho, vx, vy)
    Fcent[0,:,:] = rho * vx
    Fcent[1,:,:] = rho * vx * vx + P
    Fcent[2,:,:] = rho * vx * vy
    Fcent[3,:,:] = ( E + P ) * vx
    return Fcent

def buildGcent4(rho, vx, vy, P, gamma) :
    shape = rho.shape
    nx = shape[0]
    ny = shape[1]
    Gcent = np.zeros([4,nx,ny])
    E = getE4(P, gamma, rho, vx, vy)
    Gcent[0,:,:] = rho * vy
    Gcent[1,:,:] = rho * vx * vy
    Gcent[2,:,:] = rho * vy * vy + P
    Gcent[3,:,:] = ( E + P ) * vy
    return Gcent

def splitScalar(array) :
    length = len(array)
    arrayL = array[0:length-1]
    arrayR = array[1:length]
    return arrayL, arrayR

def splitScalarX(array) :
    shape = array.shape
    nx = shape[0]
    arrayL = array[0:nx-1, :]
    arrayR = array[1:nx, :]
    return arrayL, arrayR

def splitScalarY(array) :
    shape = array.shape
    ny = shape[1]
    arrayB = array[:, 0:ny-1]
    arrayT = array[:, 1:ny]
    return arrayB, arrayT

def splitVector(array) :
    length = array.shape[1]
    arrayL = array[:,0:length-1]
    arrayR = array[:,1:length]
    return arrayL, arrayR

def splitVectorX(array) :
    shape = array.shape
    nx = shape[1]
    arrayL = array[:, 0:nx-1, :]
    arrayR = array[:, 1:nx, :]
    return arrayL, arrayR

def splitVectorY(array) :
    shape = array.shape
    ny = shape[2]
    arrayB = array[:, :, 0:ny-1]
    arrayT = array[:, :, 1:ny]
    return arrayB, arrayT

def splitRecon(var) :
    length = len(var)
    varm1 = var[0:length-3]
    var0 = var[1:length-2]
    varp1 = var[2:length-1]
    varp2 = var[3:length]
    return varm1, var0, varp1, varp2

def splitReconX(var) :
    shape = var.shape
    nx = shape[0]
    varm1 = var[0:nx-3, :]
    var0 = var[1:nx-2, :]
    varp1 = var[2:nx-1, :]
    varp2 = var[3:nx, :]
    return varm1, var0, varp1, varp2

def splitReconY(var) :
    shape = var.shape
    ny = shape[1]
    varm1 = var[:, 0:ny-3]
    var0 = var[:, 1:ny-2]
    varp1 = var[:, 2:ny-1]
    varp2 = var[:, 3:ny]
    return varm1, var0, varp1, varp2

def flipVelX(array) :
    array[1,:,:] = -array[1,:,:]
    return array

def flipVelY(array) :
    array[2,:,:] = -array[2,:,:]
    return array

def cutError(U, threshold=1.0e-15) :
    boolArray = np.absolute(U) < threshold
    U[boolArray] = 0.0
    return U
