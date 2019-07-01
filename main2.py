# import matplotlib
# matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
import numpy as np
import argparse
import sys
from sys import stdout
from lib import *

# np.seterr( all='raise' )

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--rk', action='store_true')
parser.add_argument('--recon', action='store_true')
parser.add_argument('--mirror', action='store_true')
parser.add_argument('--corner', action='store_true')
args = parser.parse_args()

# set parameters
viewY       = 22 # index
nCellsX     = 400
nCellsY     = 400
nSteps      = 1500
mirrorCellX = 300 # index
mirrorCellY = 25
downsample  = 100
gamma       = 1.4
courantFac  = 0.5
boxSizeX    = 5.0
boxSizeY    = 5.0
xDiscont    = 2.0
yDiscont    = 2.0
P1          = 100.0
P2          = 1.0
rho1        = 10.0
rho2        = 1.0
v1          = 0.0
v2          = 0.0
period      = 200
theta       = 1.5 # 1 to 2, more diffusive for theta = 1
threshold   = 1.0e-100

# set initial conditions
cellRight = int( nCellsX * xDiscont / boxSizeX )
cellTop   = int( nCellsY * yDiscont / boxSizeY )

P = np.ones((nCellsX, nCellsY)) * P1
rho = np.ones((nCellsX, nCellsY)) * rho1
vx = np.zeros((nCellsX, nCellsY))
vy = np.zeros((nCellsX, nCellsY))

P[   cellRight:nCellsX, : ] = P2
rho[ cellRight:nCellsX, : ] = rho2

if args.corner :
    P[   :, cellTop:nCellsY ] = P2
    rho[ :, cellTop:nCellsY ] = rho2

# set positions and widths
dx = boxSizeX / float(nCellsX)
dy = boxSizeY / float(nCellsY)
x = ( np.arange(0,nCellsX) + 0.5 ) * dx
y = ( np.arange(0,nCellsY) + 0.5 ) * dy

# preallocate some arrays
cons1 = np.zeros(nSteps)
cons2 = np.zeros(nSteps)
cons3 = np.zeros(nSteps)
cons4 = np.zeros(nSteps)
rhoAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
vxAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
vyAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
PAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
tAnim = np.zeros(nSteps+1)

def getMirrorSides(U, gamma, mirrorCellX, mirrorCellY) :
    # get groups of cells on each side of reflective cell

    # cut a square around the reflective cell
    square = U[:, (mirrorCellX-1):(mirrorCellX+2), (mirrorCellY-1):(mirrorCellY+2)][:]

    # separate into regions on all sides (1x2 or 2x1)
    left   = square[:, 0:2, 1:2].copy()
    right  = square[:, 1:3, 1:2].copy()
    bottom = square[:, 1:2, 0:2].copy()
    top    = square[:, 1:2, 1:3].copy()

    # make copies to get around dumb Python errors
    leftFlip   = left.copy()
    rightFlip  = right.copy()
    bottomFlip = bottom.copy()
    topFlip    = top.copy()

    leftTot   = left.copy()
    rightTot  = right.copy()
    bottomTot = bottom.copy()
    topTot    = top.copy()

    # flip velocities inside ghost cells
    leftFlip   = flipVelX(leftFlip)
    rightFlip  = flipVelX(rightFlip)
    bottomFlip = flipVelY(bottomFlip)
    topFlip    = flipVelY(topFlip)

    # put it all together
    leftTot[:, 1, :]   = leftFlip[:, 0, :]
    rightTot[:, 0, :]  = rightFlip[:, 1, :]
    bottomTot[:, :, 1] = bottomFlip[:, :, 0]
    topTot[:, :, 0]    = topFlip[:, :, 1]

    return leftTot, rightTot, bottomTot, topTot

def getMirrorSidesRecon(U, gamma, mirrorCellX, mirrorCellY) :
    # get groups of cells on each side of reflective cell

    # cut a square around the reflective cell
    square = U[:, (mirrorCellX-3):(mirrorCellX+4), (mirrorCellY-3):(mirrorCellY+4)]

    # separate into regions on all sides (1x5 or 5x1)
    # each has 3 real cells and 2 ghost ones
    left   = square[:, 0:5, 3:4].copy()
    right  = square[:, 2:7, 3:4].copy()
    bottom = square[:, 3:4, 0:5].copy()
    top    = square[:, 3:4, 2:7].copy()

    # make copies to get around dumb Python errors
    leftFlip   = left.copy()
    rightFlip  = right.copy()
    bottomFlip = bottom.copy()
    topFlip    = top.copy()

    leftTot   = left.copy()
    rightTot  = right.copy()
    bottomTot = bottom.copy()
    topTot    = top.copy()

    # flip velocities inside ghost cells
    leftFlip   = flipVelX(leftFlip[:, 1:3, :])
    rightFlip  = flipVelX(rightFlip[:, 2:4, :])
    bottomFlip = flipVelY(bottomFlip[:, :, 1:3])
    topFlip    = flipVelY(topFlip[:, :, 2:4])

    # put it all together
    leftTot[:, 3:5, :]   = leftFlip
    rightTot[:, 0:2, :]  = rightFlip
    bottomTot[:, :, 3:5] = bottomFlip
    topTot[:, :, 0:2]    = topFlip

    return leftTot, rightTot, bottomTot, topTot

def getMirrorFlux(U, gamma, mirrorCellX, mirrorCellY) :
    # get fluxes around the reflective cell

    # get clumps of cells on all sides of mirror
    left, right, bottom, top = getMirrorSides(U, gamma, mirrorCellX, mirrorCellY)

    # get fluxes from each clump
    FfaceL, GfaceL, alphaMaxL = getFlux(left, gamma)
    FfaceR, GfaceR, alphaMaxR = getFlux(right, gamma)
    FfaceB, GfaceB, alphaMaxB = getFlux(bottom, gamma)
    FfaceT, GfaceT, alphaMaxT = getFlux(top, gamma)

    # find the maximum mirror alpha
    alphaMax = max( [alphaMaxL, alphaMaxR, alphaMaxB, alphaMaxT] )

    return FfaceL, FfaceR, GfaceB, GfaceT, alphaMax

def getMirrorFluxRecon(U, gamma, mirrorCellX, mirrorCellY) :
    # get fluxes around the reflective cell

    # get clumps of cells on all sides of mirror
    left, right, bottom, top = getMirrorSidesRecon(U, gamma, mirrorCellX, mirrorCellY)

    # get fluxes from each clump
    FfaceL, GfaceL, alphaMaxL = getFluxRecon(left, gamma)
    FfaceR, GfaceR, alphaMaxR = getFluxRecon(right, gamma)
    FfaceB, GfaceB, alphaMaxB = getFluxRecon(bottom, gamma)
    FfaceT, GfaceT, alphaMaxT = getFluxRecon(top, gamma)

    # find the maximum mirror alpha
    alphaMax = max( [alphaMaxL, alphaMaxR, alphaMaxB, alphaMaxT] )

    return FfaceL, FfaceR, GfaceB, GfaceT, alphaMax

def getEdgeStates(U, gamma) :
    # get the lines of cells around the edges of the box

    shape = U.shape
    nx = shape[1]
    ny = shape[2]

    # separate the edge slices
    UL = U[:, 0:1, :].copy()
    UR = U[:, (nx-1):nx, :].copy()
    UB = U[:, :, 0:1].copy()
    UT = U[:, :, (ny-1):ny].copy()

    # make some copies to be flipped
    ULflip = UL.copy()
    URflip = UR.copy()
    UBflip = UB.copy()
    UTflip = UT.copy()

    # flip the velocities in the ghost cells
    ULflip = flipVelX(ULflip)
    URflip = flipVelX(URflip)
    UBflip = flipVelY(UBflip)
    UTflip = flipVelY(UTflip)

    # put it all together
    UL2 = np.zeros((4, 2, ny))
    UL2[:, 0, :] = ULflip[:, 0, :]
    UL2[:, 1, :] = UL[:, 0, :]

    UR2 = np.zeros((4, 2, ny))
    UR2[:, 0, :] = UR[:, 0, :]
    UR2[:, 1, :] = URflip[:, 0, :]

    UB2 = np.zeros((4, nx, 2))
    UB2[:, :, 0] = UBflip[:, :, 0]
    UB2[:, :, 1] = UB[:, :, 0]

    UT2 = np.zeros((4, nx, 2))
    UT2[:, :, 0] = UT[:, :, 0]
    UT2[:, :, 1] = UTflip[:, :, 0]

    return UL2, UR2, UB2, UT2

def getEdgeStatesRecon(U, gamma) :
    # get the lines of cells around the edges of the box

    shape = U.shape
    nx = shape[1]
    ny = shape[2]

    # separate the edge slices
    UL = U[:, 0:3, :].copy()
    UR = U[:, (nx-3):nx, :].copy()
    UB = U[:, :, 0:3].copy()
    UT = U[:, :, (ny-3):ny].copy()

    # make some copies to be flipped
    ULflip = UL.copy()
    URflip = UR.copy()
    UBflip = UB.copy()
    UTflip = UT.copy()

    # flip the velocities in the ghost cells
    ULflip = flipVelX(ULflip)
    URflip = flipVelX(URflip)
    UBflip = flipVelY(UBflip)
    UTflip = flipVelY(UTflip)

    # put it all together
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

    # extract state variables
    rho, vx, vy, P = getState4(U, gamma)

    # get cell-centered fluxes
    Fcent = buildFcent4(rho, vx, vy, P, gamma)
    Gcent = buildGcent4(rho, vx, vy, P, gamma)

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

    # find the maximum alpha (if it's defined)
    np.seterr( all='raise' )
    try :
        alphaMaxX = np.maximum( alphaPx.max(), alphaMx.max() )
    except :
        alphaMaxX = 0.0
    try :
        alphaMaxY = np.maximum( alphaPy.max(), alphaMy.max() )
    except :
        alphaMaxY = 0.0
    np.seterr( all='ignore' )

    alphaMax = np.maximum(alphaMaxX, alphaMaxY)

    # find fluxes at faces
    Fface = ( alphaPx * FcentL + alphaMx * FcentR - alphaPx * alphaMx * (UR - UL) ) / ( alphaPx + alphaMx )
    Gface = ( alphaPy * GcentB + alphaMy * GcentT - alphaPy * alphaMy * (UT - UB) ) / ( alphaPy + alphaMy )

    return Fface, Gface, alphaMax

def getFluxRecon(U, gamma) :

    # extract state variables
    rho, vx, vy, P = getState4(U, gamma)

    # split state variables for reconstruction
    rhom1x, rho0x, rhop1x, rhop2x = splitReconX(rho)
    vxm1x, vx0x, vxp1x, vxp2x = splitReconX(vx)
    vym1x, vy0x, vyp1x, vyp2x = splitReconX(vy)
    Pm1x, P0x, Pp1x, Pp2x = splitReconX(P)

    rhom1y, rho0y, rhop1y, rhop2y = splitReconY(rho)
    vxm1y, vx0y, vxp1y, vxp2y = splitReconY(vx)
    vym1y, vy0y, vyp1y, vyp2y = splitReconY(vy)
    Pm1y, P0y, Pp1y, Pp2y = splitReconY(P)

    # do the reconstruction
    rhoL, rhoR = reconstruct(rhom1x, rho0x, rhop1x, rhop2x, theta)
    vxL, vxR = reconstruct(vxm1x, vx0x, vxp1x, vxp2x, theta)
    vyL, vyR = reconstruct(vym1x, vy0x, vyp1x, vyp2x, theta)
    PL, PR = reconstruct(Pm1x, P0x, Pp1x, Pp2x, theta)

    rhoB, rhoT = reconstruct(rhom1y, rho0y, rhop1y, rhop2y, theta)
    vxB, vxT = reconstruct(vxm1y, vx0y, vxp1y, vxp2y, theta)
    vyB, vyT = reconstruct(vym1y, vy0y, vyp1y, vyp2y, theta)
    PB, PT = reconstruct(Pm1y, P0y, Pp1y, Pp2y, theta)

    # remake U and Fcent with reconstructed variables
    UL = buildU4(rhoL, vxL, vyL, PL, gamma)
    UR = buildU4(rhoR, vxR, vyR, PR, gamma)
    UB = buildU4(rhoB, vxB, vyB, PB, gamma)
    UT = buildU4(rhoT, vxT, vyT, PT, gamma)
    FcentL = buildFcent4(rhoL, vxL, vyL, PL, gamma)
    FcentR = buildFcent4(rhoR, vxR, vyR, PR, gamma)
    GcentB = buildGcent4(rhoB, vxB, vyB, PB, gamma)
    GcentT = buildGcent4(rhoT, vxT, vyT, PT, gamma)

    # get sound speed
    cL = getc(gamma, PL, rhoL)
    cR = getc(gamma, PR, rhoR)
    cB = getc(gamma, PB, rhoB)
    cT = getc(gamma, PT, rhoT)

    # find eigenvalues
    lambdaPL = vxL + cL
    lambdaPR = vxR + cR
    lambdaML = vxL - cL
    lambdaMR = vxR - cR
    alphaPx = max3( lambdaPL, lambdaPR )
    alphaMx = max3( -lambdaML, -lambdaMR )

    lambdaPB = vyB + cB
    lambdaPT = vyT + cT
    lambdaMB = vyB - cB
    lambdaMT = vyT - cT
    alphaPy = max3( lambdaPB, lambdaPT )
    alphaMy = max3( -lambdaMB, -lambdaMT )

    # find the maximum alpha (if it's defined)
    np.seterr( all='raise' )
    try :
        alphaMaxX = np.maximum( alphaPx.max(), alphaMx.max() )
    except :
        alphaMaxX = 0.0
    try :
        alphaMaxY = np.maximum( alphaPy.max(), alphaMy.max() )
    except :
        alphaMaxY = 0.0
    np.seterr( all='ignore' )

    alphaMax = np.maximum(alphaMaxX, alphaMaxY)

    # find face fluxes
    Fface = ( alphaPx * FcentL + alphaMx * FcentR - alphaPx * alphaMx * (UR - UL) ) / ( alphaPx + alphaMx )
    Gface = ( alphaPy * GcentB + alphaMy * GcentT - alphaPy * alphaMy * (UT - UB) ) / ( alphaPy + alphaMy )

    return Fface, Gface, alphaMax

def getL(U, gamma, dx, dy) :
    shape = U.shape
    nCellsX = shape[1]
    nCellsY = shape[2]

    # get flux on interior faces
    FfaceI, GfaceI, alphaMaxI = getFlux(U, gamma)

    # get flux on edges
    UL, UR, UB, UT = getEdgeStates(U, gamma)
    FfaceL, GfaceL, alphaMaxL = getFlux(UL, gamma)
    FfaceR, GfaceR, alphaMaxR = getFlux(UR, gamma)
    FfaceB, GfaceB, alphaMaxB = getFlux(UB, gamma)
    FfaceT, GfaceT, alphaMaxT = getFlux(UT, gamma)

    # construct the full flux arrays
    FfaceFull = np.zeros((4, nCellsX+1, nCellsY))
    GfaceFull = np.zeros((4, nCellsX, nCellsY+1))

    FfaceFull[:, 1:nCellsX, :] = FfaceI
    GfaceFull[:, :, 1:nCellsY] = GfaceI
    FfaceFull[:, 0, :]         = FfaceL[:,0,:]
    FfaceFull[:, nCellsX, :]   = FfaceR[:,0,:]
    GfaceFull[:, :, 0]         = GfaceB[:,:,0]
    GfaceFull[:, :, nCellsY]   = GfaceT[:,:,0]

    # find the overall largest alpha
    alphaMax = max( [alphaMaxI, alphaMaxL, alphaMaxR, alphaMaxB, alphaMaxT] )

    if args.mirror :
        # get the fluxes from the sides of the mirror cell
        FfaceL_mirror, FfaceR_mirror, GfaceB_mirror, GfaceT_mirror, alphaMax_mirror = getMirrorFlux(U, gamma, mirrorCellX, mirrorCellY)
        FfaceFull[:, mirrorCellX, mirrorCellY]   = FfaceL_mirror[:, 0, 0]
        FfaceFull[:, mirrorCellX+1, mirrorCellY] = FfaceR_mirror[:, 0, 0]
        GfaceFull[:, mirrorCellX, mirrorCellY]   = GfaceB_mirror[:, 0, 0]
        GfaceFull[:, mirrorCellX, mirrorCellY+1] = GfaceT_mirror[:, 0, 0]

        alphaMax = max( [alphaMax, alphaMax_mirror] )

    # cut the error from the fluxes
    # FfaceFull = cutError(FfaceFull)
    # GfaceFull = cutError(GfaceFull)

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
    UL, UR, UB, UT = getEdgeStatesRecon(U, gamma)
    FfaceL, GfaceL, alphaMaxL = getFluxRecon(UL, gamma)
    FfaceR, GfaceR, alphaMaxR = getFluxRecon(UR, gamma)
    FfaceB, GfaceB, alphaMaxB = getFluxRecon(UB, gamma)
    FfaceT, GfaceT, alphaMaxT = getFluxRecon(UT, gamma)

    # construct the full flux arrays
    FfaceFull = np.zeros((4, nCellsX+1, nCellsY))
    GfaceFull = np.zeros((4, nCellsX, nCellsY+1))

    FfaceFull[:, 2:(nCellsX-1), :]           = FfaceI
    GfaceFull[:, :, 2:(nCellsY-1)]           = GfaceI
    FfaceFull[:, 0:2, :]                     = FfaceL[:, 1:3, :]
    FfaceFull[:, (nCellsX-1):(nCellsX+1), :] = FfaceR[:, 0:2, :]
    GfaceFull[:, :, 0:2]                     = GfaceB[:, :, 1:3]
    GfaceFull[:, :, (nCellsY-1):(nCellsY+1)] = GfaceT[:, :, 0:2]

    # find the overall largest alpha
    alphaMax = max( [alphaMaxI, alphaMaxL, alphaMaxR, alphaMaxB, alphaMaxT] )

    if args.mirror :
        # get the fluxes from the sides of the mirror cell
        FfaceL_mirror, FfaceR_mirror, GfaceB_mirror, GfaceT_mirror, alphaMax_mirror = getMirrorFluxRecon(U, gamma, mirrorCellX, mirrorCellY)
        FfaceFull[:, (mirrorCellX-1):(mirrorCellX+1), mirrorCellY] = FfaceL_mirror[:,:,0]
        FfaceFull[:, (mirrorCellX+1):(mirrorCellX+3), mirrorCellY] = FfaceR_mirror[:,:,0]
        GfaceFull[:, mirrorCellX, (mirrorCellY-1):(mirrorCellY+1)] = GfaceB_mirror[:,0,:]
        GfaceFull[:, mirrorCellX, (mirrorCellY+1):(mirrorCellY+3)] = GfaceT_mirror[:,0,:]

        alphaMax = max( [alphaMax, alphaMax_mirror] )

    # cut the error from the fluxes
    # FfaceFull = cutError(FfaceFull)
    # GfaceFull = cutError(GfaceFull)

    # split flux arrays
    FfaceFullL, FfaceFullR = splitVectorX(FfaceFull)
    GfaceFullB, GfaceFullT = splitVectorY(GfaceFull)

    # find time derivatives
    L = - ( FfaceFullR - FfaceFullL ) / dx - ( GfaceFullT - GfaceFullB ) / dy

    return L, alphaMax

# initialize t, U
t = 0.0
U = buildU4(rho, vx, vy, P, gamma)
rho, vx, vy, P = getState4(U, gamma)

# save variables for animation
rhoAnim[0,:,:] = rho
vxAnim[0,:,:] = vx
vyAnim[0,:,:] = vy
PAnim[0,:,:] = P
tAnim[0] = t

for i in range(0,nSteps) :

    stdout.write( '\r t = ' + str(t)[0:5] )
    stdout.flush()

    # conserved variables
    # cons1[i], cons2[i], cons3[i], cons4[i] = getCons4(U)

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
        # use 3rd-order RK time integration

        U1 = U + minStep * L

        # clean up U1
        U1 = cutError(U1, threshold)
        U1 = smoothMirror(U1, args.mirror, mirrorCellX, mirrorCellY)

        # do Riemann solve
        if args.recon :
            L1, alphaMax1 = getLRecon(U1, gamma, dx, dy)
        else :
            L1, alphaMax1 = getL(U1, gamma, dx, dy)

        U2 = 0.75 * U + 0.25 * U1 + 0.25 * minStep * L1

        # clean up U2
        U2 = cutError(U2, threshold)
        U2 = smoothMirror(U2, args.mirror, mirrorCellX, mirrorCellY)

        # do Riemann solve
        if args.recon :
            L2, alphaMax2 = getLRecon(U2, gamma, dx, dy)
        else :
            L2, alphaMax2 = getL(U2, gamma, dx, dy)

        UNew = 1./3. * U + 2./3. * U2 + 2./3. * minStep * L2

    else :
        # 1st-order time integration
        UNew = U + minStep * L

    U = UNew

    # clean up U
    U = cutError(U, threshold)
    U = smoothMirror(U, args.mirror, mirrorCellX, mirrorCellY)

    # tease out new state variables
    rho, vx, vy, P = getState4(U, gamma)
    t = t + minStep

    # save variables for animation
    rhoAnim[i+1,:,:] = rho
    vxAnim[i+1,:,:] = vx
    vyAnim[i+1,:,:] = vy
    PAnim[i+1,:,:] = P
    tAnim[i+1] = t

stdout.write('\nDone crunching numbers\n')

# downsample timesteps for plotting
rhoAnim = rhoAnim[::downsample,:,:]
vxAnim = vxAnim[::downsample,:,:]
vyAnim = vyAnim[::downsample,:,:]
PAnim = PAnim[::downsample,:,:]
tAnim = tAnim[::downsample]
nFrames = len(tAnim)

# find some plot limits
rhoMin = rhoAnim.min()
rhoMax = rhoAnim.max()
vxMin = vxAnim.min()
vxMax = vxAnim.max()
vyMin = vyAnim.min()
vyMax = vyAnim.max()
PMin = PAnim.min()
PMax = PAnim.max()

# animate one slice (in y) of the output
plt.clf()
fig = plt.figure(figsize=(9,9))
def animate(i) :
    plt.clf()

    plt.subplot(2,2,1)
    plt.scatter(x,rhoAnim[i,:,viewY],s=1)
    plt.axis([0.01,0.01+boxSizeX,0.0,rhoMax])
    plt.axvline( x=x[mirrorCellX], c='k' )
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('t = ' + str(tAnim[i]))

    plt.subplot(2,2,2)
    plt.scatter(x,PAnim[i,:,viewY],s=1)
    plt.axis([0.01,0.01+boxSizeX,0.0,PMax])
    plt.axvline( x=x[mirrorCellX], c='k' )
    plt.xlabel('x')
    plt.ylabel('Pressure')

    plt.subplot(2,2,3)
    plt.scatter(x,vxAnim[i,:,viewY],s=1)
    plt.axis([0.01,0.01+boxSizeX,vxMin,vxMax])
    plt.axvline( x=x[mirrorCellX], c='k' )
    plt.xlabel('x')
    plt.ylabel('x Velocity')

    plt.subplot(2,2,4)
    plt.scatter(x,vyAnim[i,:,viewY],s=1)
    # plt.axis([0.01,0.01+boxSizeX,vyMin,vyMax])
    plt.axis([0.01,0.01+boxSizeX,-10.0,10.0])
    plt.axvline( x=x[mirrorCellX], c='k' )
    plt.xlabel('x')
    plt.ylabel('y Velocity')

    plt.tight_layout()

anim = animation.FuncAnimation(fig, animate, frames = nFrames, interval = period, repeat = False)
saveas = 'hydrout.mp4'
anim.save(saveas)
print('Saved animation ' + saveas)

plt.clf()

def makeAnim(name, varAnim, varMin, varMax, nFrames, period) :
    plt.clf()
    varAnim = np.transpose( varAnim, (0,2,1) )
    fig = plt.figure(figsize=(9,9))
    ims = []
    for i in range(0,nFrames) :
        im = plt.imshow( varAnim[i,:,:], vmin=varMin, vmax=varMax, interpolation='none', origin='lower' )
        ims.append([im])
    anim = animation.ArtistAnimation(fig, ims, interval = period, repeat = False)
    saveas = name + '.mp4'
    anim.save(saveas)
    plt.clf()

# animate 2-D histograms of the output
makeAnim('rhohist', rhoAnim, rhoMin, rhoMax, nFrames, period)
makeAnim('vxhist',  vxAnim,  vxMin,  vxMax,  nFrames, period)
makeAnim('vyhist',  vyAnim,  vyMin,  vyMax,  nFrames, period)
makeAnim('Phist',   PAnim,   PMin,   PMax,   nFrames, period)

print('Saved histograms')
