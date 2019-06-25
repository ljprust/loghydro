import numpy as np

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

def getCons3(U) :
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

def flipVelScalarX(U, gamma) :
    rho, vx, vy, P = getStateArray(U, gamma)
    vx = -vx
    Uflip = buildUArray(rho, vx, vy, P, gamma)
    return Uflip

def flipVelScalarY(U, gamma) :
    rho, vx, vy, P = getStateArray(U, gamma)
    vy = -vy
    ULflip = buildUArray(rho, vx, vy, P, gamma)
    return ULflip

def flipVelVectorX(U, gamma) :
    rho, vx, vy, P = getState4(U, gamma)
    vx = -vx
    Uflip = buildU4(rho, vx, vy, P, gamma)
    return Uflip

def flipVelVectorY(U, gamma) :
    rho, vx, vy, P = getState4(U, gamma)
    vy = -vy
    Uflip = buildU4(rho, vx, vy, P, gamma)
    return Uflip

def cutError(U, threshold=1.0e-15) :
    boolArray = np.absolute(U) < threshold
    U[boolArray] = 0.0
    return U
