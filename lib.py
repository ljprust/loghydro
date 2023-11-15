import numpy as np
import scipy.special as ss

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

def reconstructGP(zT, U_stencil, hydroVariable) :
    G = U_stencil[hydroVariable,:]
    fstarbar = np.matmul(zT,G)
    return fstarbar

def RiemannGP(U, gamma, deltax, zTL, zTR, Rstencil) :
    nCells = len(deltax)
    nFaces = nCells-1

    # extract cell-centered state variables
    rho, v, P = getState3(U, gamma)

    # initialize left and right states assuming 1st spatial order
    rhoL, rhoR = splitScalar(rho)
    vL, vR     = splitScalar(v)
    PL, PR     = splitScalar(P)

    for j in range(Rstencil, nFaces-Rstencil) :
        U_stencilL = U[:,j-Rstencil:j+Rstencil+1]
        U_stencilR = U[:,j-Rstencil+1:j+Rstencil+2]
        rhoL[j] = reconstructGP(zTR, U_stencilL, 0)
        rhoR[j] = reconstructGP(zTL, U_stencilR, 0)
        vL[j]   = reconstructGP(zTR, U_stencilL, 1)
        vR[j]   = reconstructGP(zTL, U_stencilR, 1)
        PL[j]   = reconstructGP(zTR, U_stencilL, 2)
        PR[j]   = reconstructGP(zTL, U_stencilR, 2)
        if (j==50) :
            print ' rho L R:',rhoL[j],rhoR[j]

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

def resetGhosts(U) :
    length = U.shape[1]

    U[0,1] =  U[0,2]
    U[1,1] = -U[1,2]
    U[2,1] =  U[2,2]

    U[0,length-2] =  U[0,length-3]
    U[1,length-2] = -U[1,length-3]
    U[2,length-2] =  U[2,length-3]

    U[0,0] =  U[0,3]
    U[1,0] = -U[1,3]
    U[2,0] =  U[2,3]

    U[0,length-1] =  U[0,length-4]
    U[1,length-1] = -U[1,length-4]
    U[2,length-1] =  U[2,length-4]

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
