import matplotlib
matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--rk', action='store_true')
parser.add_argument('--recon', action='store_true')
parser.add_argument('--mirror', action='store_true')
args = parser.parse_args()

# set parameters
gamma      = 1.4
nCells     = 10000
courantFac = 0.5
nSteps     = 30000
boxSize    = 5.0
xDiscont   = 2.0
P1         = 100.0
P2         = 1.0
rho1       = 10.0
rho2       = 1.0
v1         = 0.0
v2         = 0.0
period     = 200
downsample = 1000
theta      = 1.5 # 1 to 2, more diffusive for theta = 1
mirrorCell = 8000

# set initial conditions
cellRight = int( nCells * xDiscont / boxSize )
P = np.ones(nCells) * P1
P[ cellRight:nCells ] = P2
rho = np.ones(nCells) * rho1
rho[ cellRight:nCells ] = rho2
v = np.ones(nCells) * v1
v[ cellRight:nCells ] = v2

def addEdges(array, begin, end) :
    con = np.concatenate( ([begin],array,[end]), axis=0 )
    return con

# add ghost cells
P = addEdges(P, P[0], P[nCells-1])
rho = addEdges(rho, rho[0], rho[nCells-1])
v = addEdges(v, -v[0], -v[nCells-1])
nCells = nCells + 2

# set positions and widths
deltax = np.ones(nCells) * boxSize / float(nCells)
dx = deltax[0]
x = range(0,nCells) * dx + 0.5 * dx

# preallocate some arrays
Fface = np.zeros([3,nCells+1])
cons1 = np.zeros(nSteps)
cons2 = np.zeros(nSteps)
cons3 = np.zeros(nSteps)
rhoAnim = np.zeros([nSteps+1,nCells])
vAnim = np.zeros([nSteps+1,nCells])
PAnim = np.zeros([nSteps+1,nCells])
tAnim = np.zeros(nSteps+1)

def minmod(x, y, z) :
    result = 0.25 * np.absolute( np.sign(x) + np.sign(y) ) \
    * ( np.sign(x) + np.sign(y) ) \
    * np.minimum( np.minimum( np.absolute(x), np.absolute(y) ), np.absolute(z)  )
    return result

def reconstruct(c, cm1, cp1, cp2) :
    cL = c + 0.5 * minmod( theta*(c-cm1), 0.5*(cp1-cm1), theta*(cp1-c) )
    cR = cp1 - 0.5 * minmod( theta*(cp1-c), 0.5*(cp2-c), theta*(cp2-cp1) )
    return cL, cR

def resetGhosts(U, nCells) :
    U[0,0] = U[0,1]
    U[1,0] = -U[1,1]
    U[2,0] = U[2,1]
    U[0,nCells-1] = U[0,nCells-2]
    U[1,nCells-1] = -U[1,nCells-2]
    U[2,nCells-1] = U[2,nCells-2]
    return U

def resetMirror(U, mirrorCell) :
    if args.mirror :
        U[0,mirrorCell-1] = U[0,mirrorCell-2]
        U[1,mirrorCell-1] = -U[1,mirrorCell-2]
        U[2,mirrorCell-1] = U[2,mirrorCell-2]
        U[0,mirrorCell] = U[0,mirrorCell+1]
        U[1,mirrorCell] = -U[1,mirrorCell+1]
        U[2,mirrorCell] = U[2,mirrorCell+1]
    return U

def getE(P, gamma, rho, v) :
    e = P / (gamma - 1.) / rho
    E = rho * ( e + 0.5 * v * v )
    return E

def buildU(rho, v, E, nCells) :
    U = np.zeros([3,nCells])
    U[0,0:nCells] = rho
    U[1,0:nCells] = rho * v
    U[2,0:nCells] = E
    return U

def buildFcent(rho, v, P, E, nCells) :
    Fcent = np.zeros([3,nCells])
    Fcent[0,0:nCells] = rho * v
    Fcent[1,0:nCells] = rho * v * v + P
    Fcent[2,0:nCells] = ( E + P ) * v
    return Fcent

def getCons(U) :
    cons1 = U[0,1:(nCells-1)].sum()
    cons2 = U[1,1:(nCells-1)].sum()
    cons3 = U[2,1:(nCells-1)].sum()
    return cons1, cons2, cons3

def splitScalar(array) :
    length = len(array)
    arrayL = array[0:length-1]
    arrayR = array[1:length]
    return arrayL, arrayR

def splitVector(array) :
    length = array.shape[1]
    arrayL = array[:,0:length-1]
    arrayR = array[:,1:length]
    return arrayL, arrayR

def getc(gamma, P, rho) :
    c = np.sqrt( gamma * P / rho )
    return c

def max3(array1, array2) :
    arrayMax = np.maximum( 0., np.maximum( array1, array2 ) )
    return arrayMax

def getState(U, gamma) :
    rho = U[0,:]
    v = U[1,:] / rho
    e = U[2,:] / rho - 0.5 * v * v
    P = ( gamma - 1.0 ) * rho * e
    return rho, v, P

def Riemann(U, gamma, nCells, deltax) :
    rho, v, P = getState(U, gamma)
    E = getE(P, gamma, rho, v)
    Fcent = buildFcent(rho, v, P, E, nCells)
    UL, UR = splitVector(U)
    FcentL, FcentR = splitVector(Fcent)
    c = getc(gamma, P, rho)
    lambdaP = v + c
    lambdaM = v - c
    lambdaP_L, lambdaP_R = splitScalar(lambdaP)
    lambdaM_L, lambdaM_R = splitScalar(lambdaM)
    alphaP = max3( lambdaP_L, lambdaP_R )
    alphaM = max3( -lambdaM_L, -lambdaM_R )
    Fface = np.zeros([3,nCells+1])
    Fface[:,0] = np.array([0.0, 0.0, 0.0])
    Fface[:,nCells] = np.array([0.0, 0.0, 0.0])
    Fface[:,1:nCells] = ( alphaP * FcentL + alphaM * FcentR - alphaP * alphaM * (UR - UL) ) / ( alphaP + alphaM )
    FfaceL, FfaceR = splitVector(Fface)
    L = - ( FfaceR - FfaceL ) / deltax
    return L

t = 0.0
E = getE(P, gamma, rho, v)
U = buildU(rho, v, E, nCells)
U = resetMirror(U, mirrorCell)
rho, v, P = getState(U, gamma)

# save variables for animation
rhoAnim[0,:] = rho
vAnim[0,:] = v
PAnim[0,:] = P
tAnim[0] = t

for i in range(0,nSteps) :

    # get energy from EOS
    E = getE(P, gamma, rho, v)

    # cell-centered values
    Fcent = buildFcent(rho, v, P, E, nCells)

    # conserved variables
    # cons1[i], cons2[i], cons3[i] = getCons(U)

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

    # set fluxes at outer edges of ghost cells to something arbitrary
    Fface[:,0] = np.array([0.0, 0.0, 0.0])
    Fface[:,nCells] = np.array([0.0, 0.0, 0.0])

    # find fluxes at faces
    Fface[:,1:nCells] = ( alphaP * FcentL + alphaM * FcentR - alphaP * alphaM * (UR - UL) ) / ( alphaP + alphaM )
    FfaceL, FfaceR = splitVector(Fface)

    # find time derivatives
    L = - ( FfaceR - FfaceL ) / deltax

    # find timestep
    deltat = courantFac * deltax / alphaMax
    minStep = deltat.min()

    # propagate charges
    if args.rk :
        U1 = U + minStep * L
        U1 = resetGhosts(U1, nCells)
        U1 = resetMirror(U1, mirrorCell)
        L1 = Riemann(U1, gamma, nCells, deltax)
        U2 = 0.75 * U + 0.25 * U1 + 0.25 * minStep * L1
        U2 = resetGhosts(U2, nCells)
        U2 = resetMirror(U2, mirrorCell)
        L2 = Riemann(U2, gamma, nCells, deltax)
        UNew = 1./3. * U + 2./3. * U2 + 2./3. * minStep * L2
    else :
        UNew = U + minStep * L

    # tease out new state variables
    U = UNew
    U = resetGhosts(U, nCells)
    U = resetMirror(U, mirrorCell)
    rho, v, P = getState(U, gamma)
    t = t + minStep

    # save variables for animation
    rhoAnim[i+1,:] = rho
    vAnim[i+1,:] = v
    PAnim[i+1,:] = P
    tAnim[i+1] = t
'''
cons1diff = cons1.max() - cons1.min()
cons2diff = cons2.max() - cons2.min()
cons3diff = cons3.max() - cons3.min()
print('cons1diff:',cons1diff)
print('cons2diff:',cons2diff)
print('cons3diff:',cons3diff)
'''
print('Done crunching numbers')

rhoAnim = rhoAnim[::downsample,:]
vAnim = vAnim[::downsample,:]
PAnim = PAnim[::downsample,:]
tAnim = tAnim[::downsample]
nFrames = len(tAnim)

rhoMax = rhoAnim.max()
vMin = vAnim.min()
vMax = vAnim.max()
PMax = PAnim.max()

plt.clf()
fig = plt.figure(figsize=(9,9))

def animate(i) :
    plt.clf()

    plt.subplot(2,2,1)
    plt.scatter(x,rhoAnim[i,:],s=1)
    plt.axis([0.01,0.01+boxSize,0.0,rhoMax])
    plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('t = ' + str(tAnim[i]))

    plt.subplot(2,2,2)
    plt.scatter(x,PAnim[i,:],s=1)
    plt.axis([0.01,0.01+boxSize,0.0,PMax])
    plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Pressure')

    plt.subplot(2,2,3)
    plt.scatter(x,vAnim[i,:],s=1)
    plt.axis([0.01,0.01+boxSize,vMin,vMax])
    plt.axvline( x=x[mirrorCell-1], c='k' )
    plt.xlabel('x')
    plt.ylabel('Velocity')

    plt.tight_layout()

anim = animation.FuncAnimation(fig, animate, frames = nFrames, interval = period, repeat = False)
saveas = 'hydrout.mp4'
anim.save(saveas)
print('Saved animation ' + saveas)

plt.clf()
