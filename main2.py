# import matplotlib
# matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
import numpy as np
import argparse
from sys import stdout
from lib import *

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--rk', action='store_true')
parser.add_argument('--recon', action='store_true')
parser.add_argument('--mirror', action='store_true')
args = parser.parse_args()

# set parameters
viewY       = 0
nCellsX     = 500
nCellsY     = 100
nSteps      = 1000
mirrorCellX = 50
mirrorCellY = 50
downsample  = 20
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
theta       = 1.0 # 1 to 2, more diffusive for theta = 1
threshold   = 1.0e-15

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
cons4 = np.zeros(nSteps)
rhoAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
vxAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
vyAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
PAnim = np.zeros([nSteps+1,nCellsX,nCellsY])
tAnim = np.zeros(nSteps+1)

def getEdgeStates(U, gamma) :
    shape = U.shape
    nx = shape[1]
    ny = shape[2]

    UL = U[:, 0, :]
    UR = U[:, nx-1, :]
    UB = U[:, :, 0]
    UT = U[:, :, ny-1]

    ULflip = flipVelScalarX(UL, gamma)
    URflip = flipVelScalarX(UR, gamma)
    UBflip = flipVelScalarY(UB, gamma)
    UTflip = flipVelScalarY(UT, gamma)

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

    ULflip = flipVelVectorX(UL)
    URflip = flipVelVectorX(UR)
    UBflip = flipVelVectorY(UB)
    UTflip = flipVelVectorY(UT)

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
    FcentL = buildFcent4(rhoL, vL, PL, gamma)
    FcentR = buildFcent4(rhoR, vR, PR, gamma)
    GcentB = buildGcent4(rhoB, vB, PB, gamma)
    GcentT = buildGcent4(rhoT, vT, PT, gamma)

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
    UL, UR, UB, UT = getEdgeStates(U, gamma)
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
    cons1[i], cons2[i], cons3[i], cons4[i] = getCons4(U)

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
    U = cutError(U, threshold)
    rho, vx, vy, P = getState4(U, gamma)
    t = t + minStep

    # print('cons1',cons1[i],'cons2',cons2[i],'cons3',cons3[i],'cons4',cons4[i])

    # minrho = rho.min()
    # if minrho <= 0.0 :
    #     print('\n minrho = ',minrho)
    #     break
    # minvy = vy.min()
    # if minvy < -threshold :
    #     print('\n minvy = ',minvy)
    #     break
    # minP = P.min()
    # if minP <= 0.0 :
    #     print('\n minP = ',minP)
    #     boolArray = P < 0.0
    #     print(P)
    #     print(PAnim[i,:,:])
    #     break

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
rhoMax = rhoAnim.max()
vxMin = vxAnim.min()
vxMax = vxAnim.max()
vyMin = vyAnim.min()
vyMax = vyAnim.max()
PMax = PAnim.max()

plt.clf()
fig = plt.figure(figsize=(9,9))

def animate(i) :
    plt.clf()

    plt.subplot(2,2,1)
    plt.scatter(x,rhoAnim[i,:,viewY],s=1)
    plt.axis([0.01,0.01+boxSizeX,0.0,rhoMax])
    # plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('t = ' + str(tAnim[i]))

    plt.subplot(2,2,2)
    plt.scatter(x,PAnim[i,:,viewY],s=1)
    plt.axis([0.01,0.01+boxSizeX,0.0,PMax])
    # plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Pressure')

    plt.subplot(2,2,3)
    plt.scatter(x,vxAnim[i,:,viewY],s=1)
    plt.axis([0.01,0.01+boxSizeX,vxMin,vxMax])
    # plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('x Velocity')

    plt.subplot(2,2,4)
    plt.scatter(x,vyAnim[i,:,viewY],s=1)
    # plt.axis([0.01,0.01+boxSizeX,vyMin,vyMax])
    plt.axis([0.01,0.01+boxSizeX,-10.0,10.0])
    # plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('y Velocity')

    plt.tight_layout()

anim = animation.FuncAnimation(fig, animate, frames = nFrames, interval = period, repeat = False)
saveas = 'hydrout.mp4'
anim.save(saveas)
print('Saved animation ' + saveas)

plt.clf()
