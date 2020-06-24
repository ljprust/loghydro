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
parser.add_argument('--rk', action='store_true', help='3rd-order Runge-Kutta time integration')
parser.add_argument('--recon', action='store_true', help='2nd-order spatial reconstruction')
parser.add_argument('--mirror', action='store_true', help='set one cell as an internal boundary')
args = parser.parse_args()

# set parameters
nCells     = 1000  # number of cells
nSteps     = 1000  # number of solution steps
mirrorCell = 600   # index of internal boundary cell
downsample = 100   # downsample number of timesteps for output
gamma      = 1.4   # specific heat ratio
courantFac = 0.5   # courant factor (less than 1)
boxSize    = 5.0   # size of simulation box
xDiscont   = 2.0   # initial position of discontinuity
P1         = 100.0 # pressure on left
P2         = 1.0   # pressure on right
rho1       = 10.0  # density on left
rho2       = 1.0   # density on right
v1         = 0.0   # velocity on left
v2         = 0.0   # velocity on right
period     = 200   # period of output animations in ms
theta      = 1.5   # 1 to 2, more diffusive for theta = 1

# set initial conditions
cellRight = int( nCells * xDiscont / boxSize )
P = np.ones(nCells) * P1
P[ cellRight:nCells ] = P2
rho = np.ones(nCells) * rho1
rho[ cellRight:nCells ] = rho2
v = np.ones(nCells) * v1
v[ cellRight:nCells ] = v2

# add ghost cells
P = addEdges(P, 1.0, 1.0)
P = addEdges(P, 1.0, 1.0)
rho = addEdges(rho, 1.0, 1.0)
rho = addEdges(rho, 1.0, 1.0)
v = addEdges(v, 0.0, 0.0)
v = addEdges(v, 0.0, 0.0)
nCells = nCells + 4
mirrorCell = mirrorCell + 2

# set positions and widths
deltax = np.ones(nCells) * boxSize / float(nCells)
dx = deltax[0]
x = np.arange(0,nCells) * dx + 0.5 * dx

# preallocate some arrays
cons1 = np.zeros(nSteps)
cons2 = np.zeros(nSteps)
cons3 = np.zeros(nSteps)
rhoAnim = np.zeros([nSteps+1,nCells])
vAnim = np.zeros([nSteps+1,nCells])
PAnim = np.zeros([nSteps+1,nCells])
tAnim = np.zeros(nSteps+1)

def RiemannRecon(U, gamma, deltax) :
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

# initialize t, U
t = 0.0
U = buildU3(rho, v, P, gamma)
U = resetGhosts(U)
U = resetMirror(U, args.mirror, mirrorCell)
rho, v, P = getState3(U, gamma)

# save variables for animation
rhoAnim[0,:] = rho
vAnim[0,:] = v
PAnim[0,:] = P
tAnim[0] = t

for i in range(0,nSteps) :

    stdout.write( '\r t = ' + str(t)[0:5] )
    stdout.flush()

    # conserved variables
    # cons1[i], cons2[i], cons3[i] = getCons3(U, nCells)

    # do Riemann solve
    if args.recon :
        L, alphaMax = RiemannRecon(U, gamma, deltax)
    else :
        L, alphaMax = Riemann(U, gamma, deltax)

    # find timestep
    deltat = courantFac * deltax / alphaMax
    minStep = deltat.min()

    # propagate charges
    if args.rk :
        # use 3rd-order RK time integration

        U1 = U + minStep * L

        # reset U1
        U1 = resetGhosts(U1)
        U1 = resetMirror(U1, args.mirror, mirrorCell)

        # do Riemann solve
        if args.recon :
            L1, alphaMax1 = RiemannRecon(U1, gamma, deltax)
        else :
            L1, alphaMax1 = Riemann(U1, gamma, deltax)

        # reset U2
        U2 = 0.75 * U + 0.25 * U1 + 0.25 * minStep * L1
        U2 = resetGhosts(U2)
        U2 = resetMirror(U2, args.mirror, mirrorCell)

        # do Riemann solve
        if args.recon :
            L2, alphaMax2 = RiemannRecon(U2, gamma, deltax)
        else :
            L2, alphaMax2 = Riemann(U2, gamma, deltax)

        UNew = 1./3. * U + 2./3. * U2 + 2./3. * minStep * L2

    else :
        UNew = U + minStep * L

    # tease out new state variables
    U = UNew
    U = resetGhosts(U)
    U = resetMirror(U, args.mirror, mirrorCell)
    rho, v, P = getState3(U, gamma)
    t = t + minStep

    # save variables for animation
    rhoAnim[i+1,:] = rho
    vAnim[i+1,:] = v
    PAnim[i+1,:] = P
    tAnim[i+1] = t

stdout.write('\nDone crunching numbers\n')
stdout.write('Writing output...\n')

# downsample timesteps for plotting
rhoAnim = rhoAnim[::downsample,:]
vAnim = vAnim[::downsample,:]
PAnim = PAnim[::downsample,:]
tAnim = tAnim[::downsample]
nFrames = len(tAnim)

# find some plot limits
rhoMax = rhoAnim.max()
vMin = vAnim.min()
vMax = vAnim.max()
PMax = PAnim.max()

# animate the output
plt.clf()
fig = plt.figure(figsize=(9,9))
def animate(i) :
    plt.clf()

    plt.subplot(2,2,1)
    plt.scatter(x,rhoAnim[i,:],s=1)
    plt.axis([0.01,0.01+boxSize,0.0,rhoMax])
    if (args.mirror):
        plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('t = ' + str(tAnim[i]))

    plt.subplot(2,2,2)
    plt.scatter(x,PAnim[i,:],s=1)
    plt.axis([0.01,0.01+boxSize,0.0,PMax])
    if (args.mirror):
        plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Pressure')

    plt.subplot(2,2,3)
    plt.scatter(x,vAnim[i,:],s=1)
    plt.axis([0.01,0.01+boxSize,vMin,vMax])
    if (args.mirror):
        plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Velocity')

    plt.tight_layout()

anim = animation.FuncAnimation(fig, animate, frames = nFrames, interval = period, repeat = False)
saveas = 'output.mp4'
anim.save(saveas)
print('Saved animation ' + saveas)

plt.clf()
