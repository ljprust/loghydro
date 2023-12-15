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
parser.add_argument('--gp', action='store_true', help='use GP reconstruction')
args = parser.parse_args()

# set parameters
nCells     = 800  # number of cells
nSteps     = 500  # number of solution steps
mirrorCell = 1000   # index of internal boundary cell
downsample = 50  # downsample number of timesteps for output
gamma      = 5.0/3.0   # specific heat ratio
courantFac = 0.3   # courant factor (less than 1)
boxSize    = 2.0   # size of simulation box
period     = 200   # period of output animations in ms
theta      = 1.5   # 1 to 2, more diffusive for theta = 1 (2nd order recon)
Nghosts    = 10    # number of ghost cells on each end

# GP parameters
Rstencil = 3 # stencil radius in # of cells
correlationLength = 0.0075 # hyperparameter l

# initialize primitive variable arrays
rho = np.ones(nCells)
v   = np.ones(nCells)
P   = np.ones(nCells)

'''
# add ghost cells
P = addEdges(P, 1.0, 1.0)
P = addEdges(P, 1.0, 1.0)
rho = addEdges(rho, 1.0, 1.0)
rho = addEdges(rho, 1.0, 1.0)
v = addEdges(v, 0.0, 0.0)
v = addEdges(v, 0.0, 0.0)
nCells = nCells + 4
mirrorCell = mirrorCell + 2
'''

# set positions and widths
deltax = np.ones(nCells) * boxSize / float(nCells)
dx = deltax[0]
x = np.arange(0,nCells) * dx + 0.5 * dx

# apply initial conditions
#rho, v, P =  shocktube(rho, v, P, nCells, boxSize, x, gamma)
rho, v, P =   gaussian(rho, v, P, nCells, boxSize, x, gamma)
#rho, v, P = linearWave(rho, v, P, nCells, boxSize, x, gamma)

# preallocate some arrays
cons1 = np.zeros(nSteps)
cons2 = np.zeros(nSteps)
cons3 = np.zeros(nSteps)
rhoAnim = np.zeros([nSteps+1,nCells])
vAnim = np.zeros([nSteps+1,nCells])
PAnim = np.zeros([nSteps+1,nCells])
tAnim = np.zeros(nSteps+1)

# initialize t, U
t = 0.0
U = buildU3(rho, v, P, gamma)
U = resetGhosts(U, Nghosts)
U = resetMirror(U, args.mirror, mirrorCell)
rho, v, P = getState3(U, gamma)

if (args.gp) :
    # initialize weight vector for GP reconstructions
    if (args.recon) :
        print 'WARNING: GP and 2nd order reconstruction both set!'
    stencilSize = Rstencil+Rstencil+1
    x_stencil = x[0:stencilSize]
    xstarL = 0.5*(x_stencil[Rstencil]+x_stencil[Rstencil-1])
    xstarR = 0.5*(x_stencil[Rstencil]+x_stencil[Rstencil+1])
    print x_stencil
    print xstarL, xstarR
    C = getCovarianceMatrix(x_stencil, dx, Rstencil, correlationLength)
    TTL = getPredictionVector(x_stencil, xstarL, dx, Rstencil, correlationLength)
    TTR = getPredictionVector(x_stencil, xstarR, dx, Rstencil, correlationLength)
    zTL = getWeightVector(C, TTL)
    zTR = getWeightVector(C, TTR)
    print 'zTL:',zTL
    print 'zTR:',zTR

# save variables for animation
rhoAnim[0,:] = rho
vAnim[0,:]   = v
PAnim[0,:]   = P
tAnim[0]     = t

for i in range(0,nSteps) :

    stdout.write( '\r t = ' + str(t)[0:5] )
    stdout.flush()

    # conserved variables
    # cons1[i], cons2[i], cons3[i] = getCons3(U, nCells)

    # do Riemann solve
    if args.recon :
        L, alphaMax = RiemannRecon(U, gamma, deltax, theta)
    elif args.gp :
        L, alphaMax = RiemannGP(U, gamma, deltax, zTL, zTR, Rstencil)
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
        U1 = resetGhosts(U1, Nghosts)
        U1 = resetMirror(U1, args.mirror, mirrorCell)

        # do Riemann solve
        if args.recon :
            L1, alphaMax1 = RiemannRecon(U1, gamma, deltax, theta)
        elif args.gp :
            L1, alphaMax1 = RiemannGP(U1, gamma, deltax, zTL, zTR, Rstencil)
        else :
            L1, alphaMax1 = Riemann(U1, gamma, deltax)

        # reset U2
        U2 = 0.75 * U + 0.25 * U1 + 0.25 * minStep * L1
        U2 = resetGhosts(U2, Nghosts)
        U2 = resetMirror(U2, args.mirror, mirrorCell)

        # do Riemann solve
        if args.recon :
            L2, alphaMax2 = RiemannRecon(U2, gamma, deltax, theta)
        elif args.gp :
            L2, alphaMax2 = RiemannGP(U2, gamma, deltax, zTL, zTR, Rstencil)
        else :
            L2, alphaMax2 = Riemann(U2, gamma, deltax)

        UNew = 1./3. * U + 2./3. * U2 + 2./3. * minStep * L2

    else :
        UNew = U + minStep * L

    # tease out new state variables
    U = UNew
    U = resetGhosts(U, Nghosts)
    U = resetMirror(U, args.mirror, mirrorCell)
    rho, v, P = getState3(U, gamma)
    t = t + minStep

    checkSolutionGaussian(rho, nCells, x, t)

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
