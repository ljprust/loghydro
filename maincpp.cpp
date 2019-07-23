#include <iostream>
#include <typeinfo>
#include <unistd.h>
#include <cmath>
using namespace std;

#define RK
#define RECON
#define MIRROR

// set parameters
const int nCells  = 9;
const int nSteps  = 10;
int    mirrorCell = 800;
int    downsample = 100;
int    period     = 200;
double gama       = 1.4;
double courantFac = 0.5;
double boxSize    = 5.0;
double xDiscont   = 2.0;
double P1         = 100.0;
double P2         = 1.0;
double rho1       = 10.0;
double rho2       = 1.0;
double v1         = 0.0;
double v2         = 0.0;
double theta      = 1.5; // 1 to 2, more diffusive for theta = 1

struct Charge {
  double rho[nCells];
  double v[nCells];
  double P[nCells];
  double row1[nCells];
  double row2[nCells];
  double row3[nCells];
  double e[nCells];
  double E[nCells];
};

struct CentFlux {
  double row1[nCells];
  double row2[nCells];
  double row3[nCells];
};

struct Deriv {
  double row1[nCells-2];
  double row2[nCells-2];
  double row3[nCells-2];
  double alphaMax;
};

Charge propagateCharge( struct Charge U, struct Deriv L, double deltat );
Charge buildCharge( double rho[], double v[], double P[] );
CentFlux buildCentFlux( struct Charge U );
void saveState( int stepCount, struct Charge U, double t );
Deriv Riemann( struct Charge U, struct CentFlux F, double deltax[] );
double max3( double a, double b );

// preallocate some arrays
double cons1[nSteps];
double cons2[nSteps];
double cons3[nSteps];
double rhoAnim[nSteps+1][nCells];
double vAnim[nSteps+1][nCells];
double PAnim[nSteps+1][nCells];
double tAnim[nSteps+1];

int main () {

  // find discontinuity
  int cellRight;
  double nCellsFloat;
  double cellRightFloat;
  double dx;
  nCellsFloat = (double) nCells - 4.0;
  cellRightFloat = nCellsFloat * xDiscont / boxSize;
  cellRight = (int) cellRightFloat;
  cellRight = cellRight + 2;
  dx = boxSize / nCellsFloat;

  // set initial conditions
  double P[nCells];
  double rho[nCells];
  double v[nCells];
  double deltax[nCells];
  double x[nCells];

  for( int i = 0; i < cellRight; i++ ) {
    P[i]      = P1;
    rho[i]    = rho1;
    v[i]      = v1;
    deltax[i] = dx;
    x[i]      = ( (double)i + 0.5 - 2.0 ) * dx;
  }
  for( int i = cellRight; i < nCells; i++ ) {
    P[i]      = P2;
    rho[i]    = rho2;
    v[i]      = v2;
    deltax[i] = dx;
    x[i]      = ( (double)i + 0.5 - 2.0 ) * dx;
  }

  // initialize t, U
  int stepCount = 0;
  double t = 0.0;
  struct Charge U;
  U = buildCharge( rho, v, P );
  // U = resetMirror(U, mirrorCell);
  saveState( stepCount, U, t );

  for( int j = 0; j < nSteps; j++ ) {

    stepCount++;

    cout << "\r t = " << t;
    fflush(stdout);

    struct CentFlux Fcent;
    Fcent = buildCentFlux( U );

    struct Deriv L;
    L = Riemann( U, Fcent, deltax );

    double deltat = courantFac * dx / L.alphaMax;

    struct Charge Unew;
    Unew = propagateCharge( U, L, deltat );

    // RESET GHOSTS
    // RESET MIRROR

    t = t + deltat;
    saveState( stepCount, Unew, t );
    U = Unew;

  }

  cout << "\n Done crunching numbers \n";

  return 0;
}

Charge buildCharge( double rho[], double v[], double P[] ) {
  struct Charge U;
  for( int i = 0; i < nCells; i++ ) {
    U.rho[i]  = rho[i];
    U.v[i]    = v[i];
    U.P[i]    = P[i];
    U.e[i]    = P[i] / (gama - 1.0) / rho[i];
    U.E[i]    = rho[i] * ( U.e[i] + 0.5 * v[i] * v[i] );
    U.row1[i] = rho[i];
    U.row2[i] = rho[i] * v[i];
    U.row3[i] = U.E[i];
  }
  return U;
}

Charge propagateCharge( struct Charge U, struct Deriv L, double deltat ) {
  struct Charge Unew;

  for( int i = 0; i < nCells; i++ ) {
    Unew.row1[i] = U.row1[i] + deltat * L.row1[i];
    Unew.row2[i] = U.row2[i] + deltat * L.row2[i];
    Unew.row3[i] = U.row3[i] + deltat * L.row3[i];

    Unew.rho[i] = Unew.row1[i];
    Unew.v[i] = Unew.row2[i] / Unew.rho[i];
    Unew.E[i] = Unew.row3[i];
    Unew.e[i] = Unew.E[i] / Unew.rho[i] - 0.5 * Unew.v[i] * Unew.v[i];
    Unew.P[i] = ( gama - 1.0 ) * Unew.rho[i] * Unew.e[i];
  }

  return Unew;
}

Deriv Riemann( struct Charge U, struct CentFlux F, double deltax[] ) {
  double c[nCells];
  double lambdaP[nCells], lambdaM[nCells];
  double lambdaPL[nCells-1], lambdaPR[nCells-1];
  double lambdaML[nCells-1], lambdaMR[nCells-1];
  double alphaP[nCells-1], alphaM[nCells-1];
  double Urow1L[nCells-1], Urow2L[nCells-1], Urow3L[nCells-1];
  double Urow1R[nCells-1], Urow2R[nCells-1], Urow3R[nCells-1];
  double Frow1L[nCells-1], Frow2L[nCells-1], Frow3L[nCells-1];
  double Frow1R[nCells-1], Frow2R[nCells-1], Frow3R[nCells-1];
  double Fface1[nCells-1], Fface2[nCells-1], Fface3[nCells-1];
  double Fface1L[nCells-2], Fface2L[nCells-2], Fface3L[nCells-2];
  double Fface1R[nCells-2], Fface2R[nCells-2], Fface3R[nCells-2];

  for(int i = 0; i < nCells; i++ ) {
    c[i] = sqrt( gama * U.P[i] / U.rho[i] );
    lambdaP[i] = U.v[i] + c[i];
    lambdaM[i] = U.v[i] - c[i];
  }

  for(int i = 0; i < (nCells-1); i++ ) {
    Urow1L[i] = U.row1[i];
    Urow2L[i] = U.row2[i];
    Urow3L[i] = U.row3[i];
    Frow1L[i] = F.row1[i];
    Frow2L[i] = F.row2[i];
    Frow3L[i] = F.row3[i];
    lambdaPL[i] = lambdaP[i];
    lambdaML[i] = lambdaM[i];
  }

  for(int i = 1; i < nCells; i++ ) {
    Urow1R[i] = U.row1[i];
    Urow2R[i] = U.row2[i];
    Urow3R[i] = U.row3[i];
    Frow1R[i] = F.row1[i];
    Frow2R[i] = F.row2[i];
    Frow3R[i] = F.row3[i];
    lambdaPR[i] = lambdaP[i];
    lambdaMR[i] = lambdaM[i];
  }

  double alphaMax = 0.0;
  double stepMax = 0.0;
  for(int i = 0; i < (nCells-1); i++ ) {
    alphaP[i] = max3( lambdaPL[i], lambdaPR[i] );
    alphaM[i] = max3( -1.0*lambdaML[i], -1.0*lambdaMR[i] );
    stepMax = max( alphaP[i], alphaM[i] );
    alphaMax = max( alphaMax, stepMax );

    Fface1[i] = ( alphaP[i] * Frow1L[i] + alphaM[i] * Frow1R[i]
                - alphaP[i] * alphaM[i] * (Urow1R[i] - Urow1L[i]) )
                / ( alphaP[i] - alphaM[i] );
    Fface2[i] = ( alphaP[i] * Frow2L[i] + alphaM[i] * Frow2R[i]
                - alphaP[i] * alphaM[i] * (Urow2R[i] - Urow2L[i]) )
                / ( alphaP[i] - alphaM[i] );
    Fface3[i] = ( alphaP[i] * Frow3L[i] + alphaM[i] * Frow3R[i]
                - alphaP[i] * alphaM[i] * (Urow3R[i] - Urow3L[i]) )
                / ( alphaP[i] - alphaM[i] );
  }

  for(int i = 0; i < (nCells-2); i++ ) {
    Fface1L[i] = Fface1[i];
    Fface2L[i] = Fface2[i];
    Fface3L[i] = Fface3[i];
  }

  for(int i = 1; i < (nCells-1); i++ ) {
    Fface1R[i] = Fface1[i];
    Fface2R[i] = Fface2[i];
    Fface3R[i] = Fface3[i];
  }

  struct Deriv L;
  L.alphaMax = alphaMax;
  for(int i = 0; i < (nCells-2); i++ ) {
    L.row1[i] = -1.0 * ( Fface1R[i] - Fface1L[i] ) / deltax[i];
    L.row2[i] = -1.0 * ( Fface2R[i] - Fface2L[i] ) / deltax[i];
    L.row3[i] = -1.0 * ( Fface3R[i] - Fface3L[i] ) / deltax[i];
  }

  return L;
}

CentFlux buildCentFlux( struct Charge U ) {
  struct CentFlux F;
  for( int i = 0; i < nCells; i++ ) {
    F.row1[i] = U.rho[i] * U.v[i];
    F.row2[i] = U.rho[i] * U.v[i] * U.v[i] + U.P[i];
    F.row3[i] = ( U.E[i] + U.P[i] ) * U.v[i];
  }
  return F;
}

void saveState( int stepCount, struct Charge U, double t ) {
  for( int i = 0; i < nCells; i++ ) {
    rhoAnim[stepCount][i] = U.rho[i];
    vAnim[stepCount][i] = U.v[i];
    PAnim[stepCount][i] = U.P[i];
    tAnim[stepCount] = t;
  }
}

double max3( double a, double b ) {
  double result = max( max(a, b), 0.0 );
  return result;
}
