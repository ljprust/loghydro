# import matplotlib
# matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
import numpy as np
import argparse
from sys import stdout

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--rk', action='store_true')
parser.add_argument('--recon', action='store_true')
parser.add_argument('--mirror', action='store_true')
args = parser.parse_args()

# set parameters
nCellsX     = 10
nCellsY     = 10
nSteps      = 40
mirrorCellX = 50
mirrorCellY = 50
downsample  = 10
gamma       = 1.4
courantFac  = 0.5
boxSizeX    = 5.0
boxSizeY    = 5.0
xDiscont    = 2.0
P1          = 100.0
P2          = 1.0
rho1        = 10.0
rho2        = 1.0
v1          = 0.0
v2          = 0.0
period      = 200
theta       = 1.5 # 1 to 2, more diffusive for theta = 1

# set initial conditions
cellRight = int( nCellsX * xDiscont / boxSizeX )
P = np.ones((nCellsX, nCellsY)) * P1
P[ cellRight:nCellsX, : ] = P2
rho = np.ones((nCellsX, nCellsY)) * rho1
rho[ cellRight:nCellsX, : ] = rho2
vx = np.ones((nCellsX, nCellsY)) * v1
vx[ cellRight:nCellsX, : ] = v2
vy = np.zeros((nCellsX, nCellsY))

# set positions and widths
dx = boxSizeX / float(nCellsX)
x = ( np.arange(0,nCellsX) + 0.5 ) * dx
dy = boxSizeY / float(nCellsY)
y = ( np.arange(0,nCellsY) + 0.5 ) * dy

# preallocate some arrays
cons1 = np.zeros(nSteps)
cons2 = np.zeros(nSteps)
cons3 = np.zeros(nSteps)
rhoAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
vxAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
vyAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
PAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
tAnim = np.zeros(nSteps+1)

def minmod(x, y, z) :
    result = 0.25 * np.absolute( np.sign(x) + np.sign(y) ) \
    * ( np.sign(x) + np.sign(y) ) \
    * np.minimum( np.minimum( np.absolute(x), np.absolute(y) ), np.absolute(z)  )
    return result

def reconstruct(cm1, c0, cp1, cp2, theta) :
    cL = c0 + 0.5 * minmod( theta*(c0-cm1), 0.5*(cp1-cm1), theta*(cp1-c0) )
    cR = cp1 - 0.5 * minmod( theta*(cp1-c0), 0.5*(cp2-c0), theta*(cp2-cp1) )
    return cL, cR

def getE(P, gamma, rho, vx, vy) :
    e = P / (gamma - 1.) / rho
    E = rho * ( e + 0.5 * ( vx*vx + vy*vy ) )
    return E

def buildU(rho, vx, vy, P, gamma) :
    shape = rho.shape
    nx = shape[0]
    ny = shape[1]
    U = np.zeros([4,nx,ny])
    E = getE(P, gamma, rho, vx, vy)
    U[0,0:nx,0:ny] = rho
    U[1,0:nx,0:ny] = rho * vx
    U[2,0:nx,0:ny] = rho * vy
    U[3,0:nx,0:ny] = E
    return U

def buildFcent(rho, vx, vy, P) :
    shape = rho.shape
    nx = shape[0]
    ny = shape[1]
    Fcent = np.zeros([4,nx,ny])
    E = getE(P, gamma, rho, vx, vy)
    Fcent[0,0:nx,0:ny] = rho * vx
    Fcent[1,0:nx,0:ny] = rho * vx * vx + P
    Fcent[2,0:nx,0:ny] = rho * vx * vy
    Fcent[3,0:nx,0:ny] = ( E + P ) * vx
    return Fcent

def buildGcent(rho, vx, vy, P) :
    shape = rho.shape
    nx = shape[0]
    ny = shape[1]
    Gcent = np.zeros([4,nx,ny])
    E = getE(P, gamma, rho, vx, vy)
    Gcent[0,0:nx,0:ny] = rho * vy
    Gcent[1,0:nx,0:ny] = rho * vx * vy
    Gcent[2,0:nx,0:ny] = rho * vy * vy + P
    Gcent[3,0:nx,0:ny] = ( E + P ) * vy
    return Gcent

def getCons(U) :
    cons1 = U[0,:,:].sum()
    cons2 = U[1,:,:].sum()
    cons3 = U[2,:,:].sum()
    cons4 = U[3,:,:].sum()
    return cons1, cons2, cons3, cons4

def splitScalarX(array) :
    shape = array.shape
    nx = shape[0]
    arrayB = array[0:nx-1, :]
    arrayT = array[1:nx, :]
    return arrayB, arrayT

def splitScalarY(array) :
    shape = array.shape
    ny = shape[1]
    arrayL = array[:, 0:ny-1]
    arrayR = array[:, 1:ny]
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

def getc(gamma, P, rho) :
    c = np.sqrt( gamma * P / rho )
    return c

def max3(array1, array2) :
    arrayMax = np.maximum( 0., np.maximum( array1, array2 ) )
    return arrayMax

def getState(U, gamma) :
    rho = U[0,:,:]
    vx = U[1,:,:] / rho
    vy = U[2,:,:] / rho
    e = U[3,:,:] / rho - 0.5 * ( vx*vx + vy*vy )
    P = ( gamma - 1.0 ) * rho * e
    return rho, vx, vy, P

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

def flipVelX(U) :
    try:
        U[1,:,:] = -U[1,:,:]
    except:
        U[1,:] = -U[1,:]
    return U

def flipVelY(U) :
    try:
        U[2,:,:] = -U[2,:,:]
    except:
        U[2,:] = -U[2,:]
    return U

def getEdgeStates(U) :
    shape = U.shape
    nx = shape[1]
    ny = shape[2]

    UL = U[:, 0, :]
    UR = U[:, nx-1, :]
    UB = U[:, :, 0]
    UT = U[:, :, ny-1]

    ULflip = flipVelX(UL)
    URflip = flipVelX(UR)
    UBflip = flipVelY(UB)
    UTflip = flipVelY(UT)

    UL2 = np.zeros((4, 2, ny))
    UL2[:, 0, :] = ULflip
    UL2[:, 1, :] = UL

    UR2 = np.zeros((4, 2, ny))
    UR2[:, 0, :] = UR
    UR2[:, 1, :] = URflip

    UB2 = np.zeros((4, nx, 2))
    UB2[:, :, 0] = UBflip
    UB2[:, :, 1] = UB

    UT2 = np.zeros((4, nx, 2))
    UT2[:, :, 0] = UT
    UT2[:, :, 1] = UTflip

    return UL2, UR2, UB2, UT2

def getEdgeStatesRecon(U) :
    shape = U.shape
    nx = shape[1]
    ny = shape[2]

    UL = U[:, 0:3, :]
    UR = U[:, (nx-3):nx, :]
    UB = U[:, :, 0:3]
    UT = U[:, :, (ny-3):ny]

    ULflip = flipVelX(UL)
    URflip = flipVelX(UR)
    UBflip = flipVelY(UB)
    UTflip = flipVelY(UT)

    UL2 = np.zeros((4, 6, ny))
    UL2[:, 0:3, :] = ULflip
    UL2[:, 3:6, :] = UL

    UR2 = np.zeros((4, 6, ny))
    UR2[:, 0:3, :] = UR
    UR2[:, 3:6, :] = URflip

    UB2 = np.zeros((4, nx, 6))
    UB2[:, :, 0:3] = UBflip
    UB2[:, :, 3:6] = UB

    UT2 = np.zeros((4, nx, 6))
    UT2[:, :, 0:3] = UT
    UT2[:, :, 3:6] = UTflip

    return UL2, UR2, UB2, UT2

def getFlux(U, gamma) :
    shape = U.shape
    nCellsX = shape[1]
    nCellsY = shape[2]

    # extract state variables
    rho, vx, vy, P = getState(U, gamma)

    # get cell-centered fluxes
    Fcent = buildFcent(rho, vx, vy, P)
    Gcent = buildGcent(rho, vx, vy, P)

    # split into left and right values
    UL, UR = splitVectorX(U)
    UB, UT = splitVectorY(U)
    FcentL, FcentR = splitVectorX(Fcent)
    GcentB, GcentT = splitVectorY(Gcent)

    # get sound speed
    c = getc(gamma, P, rho)

    # find eigenvalues
    lambdaPx = vx + c
    lambdaMx = vx - c
    lambdaPy = vy + c
    lambdaMy = vy - c
    lambdaPx_L, lambdaPx_R = splitScalarX(lambdaPx)
    lambdaMx_L, lambdaMx_R = splitScalarX(lambdaMx)
    lambdaPy_B, lambdaPy_T = splitScalarY(lambdaPy)
    lambdaMy_B, lambdaMy_T = splitScalarY(lambdaMy)
    alphaPx = max3( lambdaPx_L, lambdaPx_R )
    alphaMx = max3( -lambdaMx_L, -lambdaMx_R )
    alphaPy = max3( lambdaPy_B, lambdaPy_T )
    alphaMy = max3( -lambdaMy_B, -lambdaMy_T )
    alphaMaxX = np.maximum( alphaPx.max(), alphaMx.max() )
    alphaMaxY = np.maximum( alphaPy.max(), alphaMy.max() )
    alphaMax = np.maximum(alphaMaxX, alphaMaxY)

    # find fluxes at faces
    Fface = ( alphaPx * FcentL + alphaMx * FcentR - alphaPx * alphaMx * (UR - UL) ) / ( alphaPx + alphaMx )
    Gface = ( alphaPy * GcentB + alphaMy * GcentT - alphaPy * alphaMy * (UT - UB) ) / ( alphaPy + alphaMy )

    return Fface, Gface, alphaMax

def getFluxRecon(U, gamma) :
    shape = U.shape
    nCellsX = shape[1]
    nCellsY = shape[2]

    # extract state variables
    rho, vx, vy, P = getState(U, gamma)

    # split state variables for reconstruction
    rhom1x, rho0x, rhop1x, rhop2x = splitReconX(rho)
    vm1x, v0x, vp1x, vp2x = splitReconX(v)
    Pm1x, P0x, Pp1x, Pp2x = splitReconX(P)

    rhom1y, rho0y, rhop1y, rhop2y = splitReconY(rho)
    vm1y, v0y, vp1y, vp2y = splitReconY(v)
    Pm1y, P0y, Pp1y, Pp2y = splitReconY(P)

    # do the reconstruction
    rhoL, rhoR = reconstruct(rhom1x, rho0x, rhop1x, rhop2x, theta)
    vL, vR = reconstruct(vm1x, v0x, vp1x, vp2x, theta)
    PL, PR = reconstruct(Pm1x, P0x, Pp1x, Pp2x, theta)

    rhoB, rhoT = reconstruct(rhom1y, rho0y, rhop1y, rhop2y, theta)
    vB, vT = reconstruct(vm1y, v0y, vp1y, vp2y, theta)
    PB, PT = reconstruct(Pm1y, P0y, Pp1y, Pp2y, theta)

    # remake U and Fcent with reconstructed variables
    UL = buildU(rhoL, vL, PL)
    UR = buildU(rhoR, vR, PR)
    UB = buildU(rhoB, vB, PB)
    UT = buildU(rhoT, vT, PT)
    FcentL = buildFcent(rhoL, vL, PL)
    FcentR = buildFcent(rhoR, vR, PR)
    GcentB = buildFcent(rhoB, vB, PB)
    GcentT = buildFcent(rhoT, vT, PT)

    # get sound speed
    cL = getc(gamma, PL, rhoL)
    cR = getc(gamma, PR, rhoR)
    cB = getc(gamma, PB, rhoB)
    cT = getc(gamma, PT, rhoT)

    # find eigenvalues
    lambdaPL = vL + cL
    lambdaPR = vR + cR
    lambdaML = vL - cL
    lambdaMR = vR - cR
    alphaPx = max3( lambdaPL, lambdaPR )
    alphaMx = max3( -lambdaML, -lambdaMR )
    alphaMaxX = np.maximum( alphaPx.max(), alphaMx.max() )

    lambdaPB = vB + cB
    lambdaPT = vT + cT
    lambdaMB = vB - cB
    lambdaMT = vT - cT
    alphaPy = max3( lambdaPB, lambdaPT )
    alphaMy = max3( -lambdaMB, -lambdaMT )
    alphaMaxY = np.maximum( alphaPy.max(), alphaMy.max() )

    alphaMax = np.maximum(alphaMaxX, alphaMaxY)

    # find face fluxes
    Fface = np.zeros([4,nCellsX+1])
    Fface[:, 2:(nCellsX-1), :] = ( alphaPx * FcentL + alphaMx * FcentR - alphaPx * alphaMx * (UR - UL) ) / ( alphaPx + alphaMx )
    Gface[:, :, 2:(nCellsY-1)] = ( alphaPy * GcentB + alphaMy * GcentT - alphaPy * alphaMy * (UT - UB) ) / ( alphaPy + alphaMy )

    return Fface, Gface, alphaMax

def getL(U, gamma, dx, dy) :
    shape = U.shape
    nCellsX = shape[1]
    nCellsY = shape[2]

    # get flux on interior faces
    FfaceI, GfaceI, alphaMaxI = getFlux(U, gamma)

    # get flux on edges
    UL, UR, UB, UT = getEdgeStates(U)
    FfaceL, GfaceL, alphaMaxL = getFlux(UL, gamma)
    FfaceR, GfaceR, alphaMaxR = getFlux(UR, gamma)
    FfaceB, GfaceB, alphaMaxB = getFlux(UB, gamma)
    FfaceT, GfaceT, alphaMaxT = getFlux(UT, gamma)

    # find the overall largest alpha
    alphaMax = max( [alphaMaxI, alphaMaxL, alphaMaxR, alphaMaxB, alphaMaxT] )

    # construct the full flux arrays
    FfaceFull = np.zeros((4, nCellsX+1, nCellsY))
    GfaceFull = np.zeros((4, nCellsX, nCellsY+1))

    FfaceFull[:, 1:nCellsX, :] = FfaceI
    GfaceFull[:, :, 1:nCellsY] = GfaceI
    FfaceFull[:, 0, :]         = FfaceL[:,0,:]
    FfaceFull[:, nCellsX, :]   = FfaceR[:,0,:]
    GfaceFull[:, :, 0]         = GfaceB[:,:,0]
    GfaceFull[:, :, nCellsY]   = GfaceT[:,:,0]

    # split flux arrays
    FfaceFullL, FfaceFullR = splitVectorX(FfaceFull)
    GfaceFullB, GfaceFullT = splitVectorY(GfaceFull)

    # find time derivatives
    L = - ( FfaceFullR - FfaceFullL ) / dx - ( GfaceFullT - GfaceFullB ) / dy

    return L, alphaMax

def getLRecon(U, gamma, dx, dy) :
    shape = U.shape
    nCellsX = shape[1]
    nCellsY = shape[2]

    # get flux on interior faces
    FfaceI, GfaceI, alphaMaxI = getFluxRecon(U, gamma)

    # get flux on edges
    UL, UR, UB, UT = getEdgeStatesRecon(U)
    FfaceL, GfaceL, alphaMaxL = getFluxRecon(UL, gamma)
    FfaceR, GfaceR, alphaMaxR = getFluxRecon(UR, gamma)
    FfaceB, GfaceB, alphaMaxB = getFluxRecon(UB, gamma)
    FfaceT, GfaceT, alphaMaxT = getFluxRecon(UT, gamma)

    # find the overall largest alpha
    alphaMax = max( [alphaMaxI, alphaMaxL, alphaMaxR, alphaMaxB, alphaMaxT] )

    # construct the full flux arrays
    FfaceFull = np.zeros((4, nCellsX+1, nCellsY))
    GfaceFull = np.zeros((4, nCellsX, nCellsY+1))

    FfaceFull[:, 2:(nCellsX-1), :]           = FfaceI
    GfaceFull[:, :, 2:(nCellsY-1)]           = GfaceI
    FfaceFull[:, 0:2, :]                     = FfaceL[:, 3:5, :]
    FfaceFull[:, (nCellsX-1):(nCellsX+1), :] = FfaceR[:, 2:4, :]
    GfaceFull[:, :, 0:2]                     = GfaceB[:, :, 3:5]
    GfaceFull[:, :, (nCellsY-1):(nCellsY+1)] = GfaceT[:, :, 2:4]

    # split flux arrays
    FfaceFullL, FfaceFullR = splitVectorX(FfaceFull)
    GfaceFullB, GfaceFullT = splitVectorY(GfaceFull)

    # find time derivatives
    L = - ( FfaceFullR - FfaceFullL ) / dx - ( GfaceFullT - GfaceFullB ) / dy

    return L, alphaMax

# initialize t, U
t = 0.0
U = buildU(rho, vx, vy, P, gamma)
rho, vx, vy, P = getState(U, gamma)

# save variables for animation
rhoAnim[0,:] = rho
vxAnim[0,:] = vx
vyAnim[0,:] = vy
PAnim[0,:] = P
tAnim[0] = t

for i in range(0,nSteps) :

    stdout.write( '\r t = ' + str(t)[0:5] )
    stdout.flush()

    # conserved variables
    # cons1[i], cons2[i], cons3[i] = getCons(U)

    # do Riemann solve
    if args.recon :
        L, alphaMax = getLRecon(U, gamma, dx, dy)
    else :
        L, alphaMax = getL(U, gamma, dx, dy)

    # find timestep
    deltatx = courantFac * dx / alphaMax
    deltaty = courantFac * dy / alphaMax
    minStepX = deltatx.min()
    minStepY = deltaty.min()
    minStep = np.minimum(minStepX, minStepY)

    # propagate charges
    if args.rk :
        U1 = U + minStep * L

        if args.recon :
            L1, alphaMax1 = getLRecon(U1, gamma, dx, dy)
        else :
            L1, alphaMax1 = getL(U1, gamma, dx, dy)

        U2 = 0.75 * U + 0.25 * U1 + 0.25 * minStep * L1

        if args.recon :
            L2, alphaMax2 = getLRecon(U2, gamma, dx, dy)
        else :
            L2, alphaMax2 = getL(U2, gamma, dx, dy)

        UNew = 1./3. * U + 2./3. * U2 + 2./3. * minStep * L2

    else :
        UNew = U + minStep * L

    # tease out new state variables
    U = UNew
    rho, vx, vy, P = getState(U, gamma)
    t = t + minStep

    # save variables for animation
    rhoAnim[i+1,:] = rho
    vxAnim[i+1,:] = vx
    vyAnim[i+1,:] = vy
    PAnim[i+1,:] = P
    tAnim[i+1] = t

stdout.write('\nDone crunching numbers\n')

print('rho ',rho)
print('vx ',vx)
print('vy ',vy)
print('P ',P)
