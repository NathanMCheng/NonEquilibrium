#include <iostream>
#include<fstream>
#include <complex>
#include <stdio.h>
#include <ctime>
#include <sys/time.h>
#include <stdlib.h> // to allocate additional memory
#include <omp.h>
#include <algorithm>


using namespace std;
typedef complex<double> dcomp;

// Universal Constants
const dcomp I(0.0, 1.0);                                                    // imaginary unit i
const double pi = 3.1415926535897;                                  // pi
const double kB = 0.0861734;                                           // meV / K           Boltzmann Constant
const double hbar = 658.2118993;                                     // meV fs             h-bar
const double c = 2.99792458e-7;                                           // mass / fs                speed of light (in units of the lattice constant)
const double me = 5.68563e21;                                              // meV fs^2 / mass^2  electron mass

// Properties of the Superconductor
const double a = 1.0e-10;//3.82e-10;                                                 // mass                     lattice constant
const double mass = 1.9;//5.0;                                                      // 1/me                effecive mass (electron masses)
const double t0  = 3809.98174234/mass; //pow(hbar,2.0)/2/mass/me;                          // meV                 prefactor (2005.254 meV)
const double Ef = 9470.0;                                         // meV                 Fermi energy
const double Delta0 = 1.35;//20.0;                                              // meV                 gap energy (time-independent)
//const double ky = 0.0; //1.12;                                                    // a                       1/2 kF should be ?? (k_y momentum used)
//const double kF_x = sqrt((Ef/t0)-pow(ky,2.0));                             // a                       Fermi wavevector (2.173153)
const double kF = sqrt(Ef/t0);
const double hbar_wc = 48.0;                                            // meV                  cutoff energy (must be <1/2 Ef
//double W = -1773.0;                                               // meV                  interaction strength
//const double kU = sqrt((Ef+hbar_wD)/t0);                          // a                        upper wave vector limit kx
//const double kL = sqrt((Ef-hbar_wD)/t0);                            // a                        lower wave vector limit kx

// Pump Parameters
const double hbar_w0 = 3.0;           // meV                  pump energy
const double w0_p = hbar_w0/hbar;      // 1 / fs                  pump frequency (hbar_w0/hbar)
const double q0 = w0_p/c*a;        // a                       pump momentum transfer (q0 = w0/c)
const double tau_p = 100.0;        // fs                      pump full width at half maximum
const double A0 = 7.0e-8;          // (1e-8) J s / C mass  pump intensity
const double A0eff1= hbar/2.0/mass/me/a*A0*1.0e18; //	meV effective pump intensity (1) 		  e hbar A0 / (2m a)
const double A0eff2 = 1.0/2.0/mass/me*pow(A0,2.0)*1.0e36; // meV effective pump intensity (2)    e^2 A0^2 / (2m)
const double eh_2ma = hbar/2.0/mass/me/a;

// Probe Parameters
const double hbar_w0_pr = 2.5;
const double w0_pr = hbar_w0_pr/hbar;
const double q_pr = w0_pr/c*a;
const double tau_pr = 250.0;
const double A0_pr = 1.0e-8;
const double A0eff1_pr = hbar/2.0/mass/me/a*A0_pr*1.0e18;

// Define Momentum Array Variables
int Ndelta = 11; // odd
int Ndhalf = (Ndelta-1)/2; // even

const int Ntheta = 1;

const double thetaFS = 0.0;

const double h = 1.0;                                                               // fs
double tend =-157.0; // make sure to change dtmin if only looking at pump
const double dtmin = 300000.0;//250.0; (tend-dt_pr)
const double hdt = 100.0;


//double Deltak0[Ntheta];

// FS Global Variables
double kx0[Ntheta];
double kFy[Ntheta];
double kdotA[Ntheta];
double kdotA_pr[Ntheta];
int Nk[Ntheta];
int Nkrho[Ntheta];
int Nkrd[Ntheta];


int Lthetak[Ntheta];
int Ltkd[Ntheta];
double W; //[Ntheta];

int * idxkF;
double * thetakF;
int kFpoints;

// Pump Functions
dcomp Aqp(double t);                                                     //                              for positive q: complex-valued EM Hamiltonian pump parameter
dcomp Aqm(double t);                                                    //                              for negative q

// Help Functions
double deltaf(int k1, int k2);                                            //                            calculate \delta : return 1.0 if k1 = k2;

// Main Variables & Time
dcomp Delta = 0.0;
double t = 0.0; 

// Functions for the Differential Equation
// precalculates all the variables for a given k, k' and t. ie. for a single entry in each of the four matrices for a given time.
// then calculates the entry four each of the four matrices for the given time, using the following differential equations.
// finally, calcualtes and returns the value of the gap at the current time iteration
void diffEq(int itheta, double t, dcomp Delta_c, dcomp l_c[], dcomp m_c[], dcomp n_c[], dcomp o_c[]);
void diffEq2(int itheta, double t, dcomp Delta_c, dcomp l_c[], dcomp m_c[], dcomp n_c[], dcomp o_c[]);

// Current response (not yet implemented);
dcomp currentEq(double t, int idt, dcomp aDa_c_pr[], dcomp bDb_c_pr[], dcomp ab_c_pr[], dcomp aDbD_c_pr[]);

// Runge-Kutta 4th Order implementation for the Bogoliubov quasiparticle expectation values
void RungeKutta(double t, double h);
void RungeKutta2(double t, double h);

void reformDiagArrays();

//Calculate W for zero temperature
void calcW();

// Function to Calculate the Gap Function (For first and last step)
dcomp gapEq(dcomp m_c[], dcomp n_c[], dcomp o_c[]);

void printGreensVariables();

// Function to calculate the angle dependent global variables
void calcAngles();
double intToAngle(int itheta);
void arrayLengths();
double thetac;

// Function to pre-calculate L,M, terms in R and C and Delta
void preCalcs();

double get_wall_time();

// Dwave Help Functions
double dwave(double theta);

// Sort Help Functions
bool kSort(int i, int j);
void arrayIorder(int a[],  int idx[]);
void arrayDorder(double a[], int idx[]);

// Function for indexing a 3D(2D) Matrix in my 1D c++ array
int index(int itheta, int ik, int idelta);
int indexLM(int itheta, int ik, int Nq, int iq);
int indexRC(int itheta, int ik);

// Variables for  pre-calculations ////
double * R1;
double * R2;
double * C1;
double * C2;
double * C3;
double * Lp;
double * Mp;
double * Lm;
double * Mm;
double * Wk;
double * Deltak0;

// Ansatz Arrays
dcomp * l;
dcomp * m;
dcomp * n;
dcomp * o;

dcomp * l_ddt;
dcomp * m_ddt;
dcomp * n_ddt;
dcomp * o_ddt;

dcomp * l_k;
dcomp * m_k;
dcomp * n_k;
dcomp * o_k;

dcomp * l_rk;
dcomp * m_rk;
dcomp * n_rk;
dcomp * o_rk;

int indicator;

int main()
{
	int itheta, ik, i, Ntotal, Nkrhototal, Nkrdtotal, idelta;
	double tendPump, tstart;
	
	// Calculate start and end times of the Pump
	tendPump = round(tau_p/2.0*sqrt(log(1000.0)/log(2.0)));
	tstart = -tendPump;
	
	// Calculate array lengths, etc.
	printf("Calculating Arrays \n");
	calcAngles();
	
	// Calculate total length of arrays
    Ntotal = 0;
    Nkrhototal = 0;
    Nkrdtotal = 0;
    for (itheta = 0; itheta<Ntheta; itheta++){
        Ntotal += Nk[itheta];
        Nkrhototal += Nkrho[itheta];
		Nkrdtotal += Nkrd[itheta];
    }
	Ntotal = Ntotal*Ndelta;
	
	// Create my Arrays!
	l = new dcomp[Ntotal]();
	m = new dcomp[Ntotal]();
	n = new dcomp[Ntotal]();
	o = new dcomp[Ntotal]();
	
	l_k = new dcomp[Ntotal]();
	m_k = new dcomp[Ntotal]();
	n_k = new dcomp[Ntotal]();
	o_k = new dcomp[Ntotal]();
	
	l_rk = new dcomp[Ntotal]();
	m_rk = new dcomp[Ntotal]();
	n_rk = new dcomp[Ntotal]();
	o_rk = new dcomp[Ntotal]();
	
	l_ddt = new dcomp[Ntotal]();
	m_ddt = new dcomp[Ntotal]();
	n_ddt = new dcomp[Ntotal]();
	o_ddt = new dcomp[Ntotal]();
	
	// Initialize my arrays!!
	for (itheta = 0; itheta<Ntheta; itheta++){
		
		for (ik = 0; ik<Nk[itheta]; ik++){
	
			i = index(itheta, ik, Ndhalf);
			l[i] = 1.0;
			n[i] = 1.0;
		}
		
	}
	printf("%e\n",real(l[index(2,3,Ndhalf)]));
	
	// Create  pre-calculation arrays
	Wk = new double[Nkrdtotal]();
	Deltak0 = new double[Nkrdtotal]();
	
	R1 = new double[Nkrdtotal]();
	R2 = new double[Nkrdtotal]();
	C1 = new double[Nkrdtotal]();
	C2 = new double[Nkrdtotal]();
	C3 = new double[Nkrdtotal]();
	
	// To be amalgamated in a newer version!
	Lp = new double[Nkrdtotal*2]();
	Mm = new double[Nkrdtotal*2]();
	Lm = new double[Nkrdtotal*3]();
	Mp = new double[Nkrdtotal*3]();
	
	// Populate pre-calculated arrays
	// must be called before calcW
	preCalcs();
	
	// Calculate W interaction strength
    // must be called before any calls to gapEq
    calcW();
	
	// Create files for outputs
	FILE *conditionsOutput;
    conditionsOutput = fopen("Parameters.txt","w");

    FILE *arrayOut;
    arrayOut = fopen("arrayOut.txt","w");

    FILE *deltaOut;
    deltaOut = fopen("Delta.txt","w");
	
	// Initial Delta Calculation!
	Delta = gapEq(m, n, o);
    fprintf(deltaOut, "%e\t%e\t%e\n", abs(Delta),real(Delta),imag(Delta));
	
	int myIndex;
	// Print Greens Initial
	FILE *greensOut;
	greensOut = fopen("lmno.txt","w");
	for (i = 0; i<kFpoints; i++){
		myIndex = idxkF[i];
		for (idelta = 0; idelta<Ndelta; idelta++){
			fprintf(greensOut,"%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",t, thetakF[i],real(l[myIndex]),imag(l[myIndex]),real(m[myIndex]),imag(m[myIndex]),real(n[myIndex]),imag(n[myIndex]),real(o[myIndex]),imag(o[myIndex]));
			myIndex++;
		}
	}
	fclose(greensOut);
	printGreensVariables();
	
	// Write Pump & Calculation Parameters
	fprintf(conditionsOutput, "Properties of the Superconductor\n");
	fprintf(conditionsOutput, "d-wave symmetry \n");
	fprintf(conditionsOutput, "Initial gap: %f\n", abs(Delta));
	fprintf(conditionsOutput, "Interaction Strength W: %f\n", W);
	fprintf(conditionsOutput, "Fermi energy Ef: %f\n", Ef);
	fprintf(conditionsOutput, "Mass m: %f\n", mass);
	fprintf(conditionsOutput, "Lattice constant a*1e-10: %f\n", a*1.0e10);
	fprintf(conditionsOutput, "Energy cutoff: %f\n\n", hbar_wc);
	
	fprintf(conditionsOutput, "Pump Parameters \n");
	fprintf(conditionsOutput, "Pumping Fermi surface angle: %f\n", thetaFS);
	fprintf(conditionsOutput, "Pump energy: %f\n", hbar_w0);
	fprintf(conditionsOutput, "Pump tau: %f\n", tau_p);
	fprintf(conditionsOutput, "Pump Intensity A0: %e\n\n", A0);
	
	fprintf(conditionsOutput, "Probe Parameters \n");
	fprintf(conditionsOutput, "Probe energy: %f\n", hbar_w0_pr);
	fprintf(conditionsOutput, "Probe tau_pr: %f\n", tau_pr);
	fprintf(conditionsOutput, "Probe Intensity A0_pr: %e\n\n", A0_pr);
	
	fprintf(conditionsOutput, "Start Time: %f\n", tstart);
	fprintf(conditionsOutput, "End Time: %f\n", tend);
	fprintf(conditionsOutput, "Time Step Size h: %f\n", h);
	fprintf(conditionsOutput, "Probe delay time: %f\n", dtmin);
	fprintf(conditionsOutput, "Probe time step hdt: %f\n", hdt);	
	fclose(conditionsOutput);
	

	// Begin Calculation  during the pump:
	printf("During the pump \n");
	for (t = tstart; t<tendPump; t+=h){
		if(t>tend){
			break;
		}
		RungeKutta(t,h);
		printGreensVariables();
		fprintf(deltaOut, "%e\t", abs(Delta));
		fprintf(deltaOut, "%e\t", real(Delta));
		fprintf(deltaOut, "%e\n", imag(Delta)); 
	}
	// Calculation after the pump ends
	printf("After the Pump \n");
	for (t = t; t<tend; t+=h){
		RungeKutta2(t,h);
		printGreensVariables();
		fprintf(deltaOut, "%e\t", abs(Delta));
		fprintf(deltaOut, "%e\t", real(Delta));
		fprintf(deltaOut, "%e\n", imag(Delta));
	}
	printf("End of the calculation Reached");
	
 	// Print final parameters
	printf("Writing final Array Values");
	for (i=0; i<Ntotal; i++){
		fprintf(arrayOut, "%e\t", real(l[i]));
		fprintf(arrayOut, "%e\t", imag(l[i]));
		fprintf(arrayOut, "%e\t", real(m[i]));
		fprintf(arrayOut, "%e\t", imag(m[i]));
		fprintf(arrayOut, "%e\t", real(n[i]));
		fprintf(arrayOut, "%e\t", imag(n[i]));
		fprintf(arrayOut, "%e\t", real(o[i]));
		fprintf(arrayOut, "%e\n", imag(o[i]));
	} 
	
    return 0;
}

////////////////////////////////////////////////////////////////
/////			Runge-Kutta Time-stepping 					////
////////////////////////////////////////////////////////////////
void RungeKutta(double t, double h){
	int itheta;
	dcomp Delta_c;

#pragma omp parallel for schedule(dynamic, 1)
	for (itheta = 0; itheta<Ntheta; itheta++){
		int ik, idelta, i;
			
		// First R-K step:
		diffEq(itheta, t, Delta, l, m, n, o);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l_rk[i] = l[i]+h*l_ddt[i]/6.0;
				m_rk[i] = m[i]+h*m_ddt[i]/6.0;					
				n_rk[i] = n[i]+h*n_ddt[i]/6.0;
				o_rk[i] = o[i]+h*o_ddt[i]/6.0;

				l_k[i] = l[i]+h*l_ddt[i]/2.0;
				m_k[i] = m[i]+h*m_ddt[i]/2.0;					
				n_k[i] = n[i]+h*n_ddt[i]/2.0;
				o_k[i] = o[i]+h*o_ddt[i]/2.0;
			}
		}
	}
	Delta_c = gapEq(m_k,n_k,o_k);
	t += h/2.0;

	
#pragma omp parallel for schedule(dynamic,1)
	for (itheta = 0; itheta<Ntheta; itheta++){
				int ik, idelta, i;
			
		// Second R-K step:
		diffEq(itheta, t, Delta_c, l_k, m_k, n_k, o_k);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l_rk[i] += h*l_ddt[i]/3.0;
				m_rk[i] += h*m_ddt[i]/3.0;					
				n_rk[i] += h*n_ddt[i]/3.0;
				o_rk[i] += h*o_ddt[i]/3.0;

				l_k[i] = l[i]+h*l_ddt[i]/2.0;
				m_k[i] = m[i]+h*m_ddt[i]/2.0;					
				n_k[i] = n[i]+h*n_ddt[i]/2.0;
				o_k[i] = o[i]+h*o_ddt[i]/2.0;
			}
		}
	}
	Delta_c = gapEq(m_k,n_k,o_k);

#pragma omp parallel for schedule (dynamic, 1)
	for (itheta = 0; itheta<Ntheta; itheta++){
				int ik, idelta, i;
			
		// Third R-K step:
		diffEq(itheta, t, Delta_c, l_k, m_k, n_k, o_k);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l_rk[i] += h*l_ddt[i]/3.0;
				m_rk[i] += h*m_ddt[i]/3.0;					
				n_rk[i] += h*n_ddt[i]/3.0;
				o_rk[i] += h*o_ddt[i]/3.0;

				l_k[i] = l[i]+h*l_ddt[i];
				m_k[i] = m[i]+h*m_ddt[i];					
				n_k[i] = n[i]+h*n_ddt[i];
				o_k[i] = o[i]+h*o_ddt[i];
			}
		}
	}
	Delta_c = gapEq(m_k,n_k,o_k);
	t += 0.5*h;

#pragma omp parallel for schedule (dynamic, 1)
	for (itheta = 0; itheta<Ntheta; itheta++){
				int ik, idelta, i;
			
		// Fourth R-K step:
		diffEq(itheta, t, Delta_c, l_k, m_k, n_k, o_k);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l[i] = l_rk[i] + h*l_ddt[i]/6.0;
				m[i] = m_rk[i] + h*m_ddt[i]/6.0;					
				n[i] = n_rk[i] + h*n_ddt[i]/6.0;
				o[i] = o_rk[i] + h*o_ddt[i]/6.0;
			}
		}
	}
	Delta = gapEq(m,n,o);
}

void RungeKutta2(double t, double h){
	int itheta;
	dcomp Delta_c;

#pragma omp parallel for schedule(dynamic, 1)
	for (itheta = 0; itheta<Ntheta; itheta++){
		int ik, idelta, i;
			
		// First R-K step:
		diffEq2(itheta, t, Delta, l, m, n, o);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l_rk[i] = l[i]+h*l_ddt[i]/6.0;
				m_rk[i] = m[i]+h*m_ddt[i]/6.0;					
				n_rk[i] = n[i]+h*n_ddt[i]/6.0;
				o_rk[i] = o[i]+h*o_ddt[i]/6.0;

				l_k[i] = l[i]+h*l_ddt[i]/2.0;
				m_k[i] = m[i]+h*m_ddt[i]/2.0;					
				n_k[i] = n[i]+h*n_ddt[i]/2.0;
				o_k[i] = o[i]+h*o_ddt[i]/2.0;
			}
		}
	}
	Delta_c = gapEq(m_k,n_k,o_k);
	t += h/2.0;
	
#pragma omp parallel for schedule(dynamic,1)
	for (itheta = 0; itheta<Ntheta; itheta++){
				int ik, idelta, i;
			
		// Second R-K step:
		diffEq2(itheta, t, Delta_c, l_k, m_k, n_k, o_k);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l_rk[i] += h*l_ddt[i]/3.0;
				m_rk[i] += h*m_ddt[i]/3.0;					
				n_rk[i] += h*n_ddt[i]/3.0;
				o_rk[i] += h*o_ddt[i]/3.0;

				l_k[i] = l[i]+h*l_ddt[i]/2.0;
				m_k[i] = m[i]+h*m_ddt[i]/2.0;					
				n_k[i] = n[i]+h*n_ddt[i]/2.0;
				o_k[i] = o[i]+h*o_ddt[i]/2.0;
			}
		}
	}
	Delta_c = gapEq(m_k,n_k,o_k);

#pragma omp parallel for schedule (dynamic, 1)
	for (itheta = 0; itheta<Ntheta; itheta++){
				int ik, idelta, i;
			
		// Third R-K step:
		diffEq2(itheta, t, Delta_c, l_k, m_k, n_k, o_k);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l_rk[i] += h*l_ddt[i]/3.0;
				m_rk[i] += h*m_ddt[i]/3.0;					
				n_rk[i] += h*n_ddt[i]/3.0;
				o_rk[i] += h*o_ddt[i]/3.0;

				l_k[i] = l[i]+h*l_ddt[i];
				m_k[i] = m[i]+h*m_ddt[i];					
				n_k[i] = n[i]+h*n_ddt[i];
				o_k[i] = o[i]+h*o_ddt[i];
			}
		}
	}
	Delta_c = gapEq(m_k,n_k,o_k);
	t += 0.5*h;

#pragma omp parallel for schedule (dynamic, 1)
	for (itheta = 0; itheta<Ntheta; itheta++){
				int ik, idelta, i;
			
		// Fourth R-K step:
		diffEq2(itheta, t, Delta_c, l_k, m_k, n_k, o_k);
			
		for (ik = 0; ik<Nk[itheta]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(itheta, ik, idelta);
					
				l[i] = l_rk[i] + h*l_ddt[i]/6.0;
				m[i] = m_rk[i] + h*m_ddt[i]/6.0;					
				n[i] = n_rk[i] + h*n_ddt[i]/6.0;
				o[i] = o_rk[i] + h*o_ddt[i]/6.0;
			}
		}
	}
	Delta = gapEq(m,n,o);
}

/////////////////////////////////////////////////////////////////
//             Help Functions								/////
/////////////////////////////////////////////////////////////////

// Calculate the wave vector in the x direction form the integer wave number (also in the x direction)
// this is for k not k^2...!

// calculate \delta : return 1.0 if k1 = k2;
double deltaf(int k1, int k2){
    double output = 0.0;
    if (k1 == k2){
        output = 1.0;
    }

    return output;
}

////////////////////////////////////////////////////////////////
//                  Pump Pulse
////////////////////////////////////////////////////////////////
// calculated for the plus/minus delta term without A0 (included in A0eff1 and A0eff2);

// q  = +q0
dcomp Aqp(double t){

    return exp(-pow(2.0*sqrt(log(2.0))*t/tau_p,2.0))*exp(-I*w0_p*t);

}

// q = -q0
dcomp Aqm(double t){

    return exp(-pow(2.0*sqrt(log(2.0))*t/tau_p,2.0))*exp(I*w0_p*t);

}

//////////////////////////////////////////////////////////////
//                Equations of Motion                     ////
//////////////////////////////////////////////////////////////
// Calculates the equations of motions for                ////
// all the entries in the arrays while the pump is on     ////
//////////////////////////////////////////////////////////////
void diffEq(int itheta, double t, dcomp Delta_c, dcomp l_c[], dcomp m_c[], dcomp n_c[], dcomp o_c[]){
    
	// variables for various indices 
    int idelta, i, igamma, iLM2, iLM3, ikrd, irho, ik, ikrho, ikgamma, ikq, ikrdgamma, ikrddelta, i_krhodelta, i_krho, i_deltarho, iq;
	int i_krhodeltapq, i_krhodeltamq, ikgammaq;
	
	// Pump  variables
	dcomp Aqm1, Aqp1, Aqp2, Aqm2, Aqpm;
    
	// To accumulate the sums before
	double Rk;
	dcomp Ck, Deltak;
	dcomp losum, mnsum, olsum, nmsum;
    dcomp llsum, mmsum, nnsum, oosum;
	
	// Delete Later;
	double Wkc, R1c, R2c, C1c, C2c, C3c;
	
	// Helpful  terms for breaking eq up
	dcomp scterm1, scterm2, emterm1, emterm2;
	
	dcomp adagddt, addt, bdagddt, bddt;

	// Calculate the Pump at time t
	Aqm1 = Aqm(t);
	Aqp1 = Aqp(t);
    
	Aqp2 = A0eff2*pow(Aqp1,2);
    Aqm2 = A0eff2*pow(Aqm1,2);
	Aqpm = A0eff2*2.0*Aqp1*Aqm1;
	Aqm1 = kdotA[itheta]*Aqm1;
	Aqp1 = kdotA[itheta]*Aqp1;
	
	/* if (itheta==14){
		for (ik=0; ik<Nk[14]; ik++){
			for (idelta = 0; idelta<Ndelta; idelta++){
				i = index(14, ik, idelta);
				printf("l\t%d\t%e\t%e\n",idelta,real(l_c[i]),imag(l_c[i]));
				printf("m\t%d\t%e\t%e\n",idelta,real(m_c[i]),imag(m_c[i]));
				printf("n\t%d\t%e\t%e\n",idelta,real(n_c[i]),imag(n_c[i]));
				printf("o\t%d\t%e\t%e\n",idelta,real(o_c[i]),imag(o_c[i]));
			}
		}
	} */
	// Beginning  of the loop through  Nk x N_\rho
    for (ik = 0; ik<Nk[itheta]; ik++){
		for (irho = 0; irho<Ndelta; irho++){
			ikrho = index(itheta, ik, irho);
			i_krho = ik+irho;
			// Set sums to zero
			
			adagddt = 0.0;
			addt = 0.0;
			bdagddt = 0.0;
			bddt = 0.0;
		
			
			// Accumulate all the sums  over \delta, \eta
			for (idelta = 0; idelta<Ndelta; idelta++){ // sum over \delta (replace \eta by \delta for notation equivalency)
				
				i_deltarho = idelta-irho; // \delta-\rho
					
				// Foor indexing at ikrd = k'+\rho-\delta 
				i_krhodelta = i_krho-idelta; // first term used in indexing l,m,n,o
				
				// index for k' + rho - delta in R,C,L,M
				ikrd = indexRC(itheta, i_krhodelta+2*Ndhalf); // i = k ' + \rho - \delta;
				
				ikrddelta = index(itheta, i_krhodelta, idelta); // index for k'+\rho-\delta,\delta for l,m,n,o

				if(i_krhodelta>=0 && i_krhodelta<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"
					
					
					// Delta at current t
					Deltak=Wk[ikrd]*Delta_c; 
										
					Rk = R1[ikrd]+R2[ikrd]*real(Deltak);
					Ck = C1[ikrd]+C2[ikrd]*Deltak-C3[ikrd]*conj(Deltak);
										
					// Set Sums to zero
					llsum = 0.0;
					mmsum = 0.0;
					nnsum = 0.0;
					oosum = 0.0;
					losum = 0.0;
					mnsum = 0.0;
					olsum = 0.0;
					nmsum = 0.0;
					
					for(igamma = 0; igamma<Ndelta; igamma++){ // most inner  sum over \gamma
						// SC TERM CONDITIONAL
						if (abs(irho-idelta-igamma+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
							ikgamma = index(itheta, ik, igamma); // index at k',\gamma
							ikrdgamma = index(itheta,i_krhodelta,igamma+i_deltarho); // index for k'+\rho-\delta,\gamma+\delta-\rho
							
							
							// Accumulate sums over gamma
							llsum += l_c[ikgamma]*conj(l_c[ikrdgamma]);
							mmsum += m_c[ikgamma]*conj(m_c[ikrdgamma]);
							nnsum += n_c[ikgamma]*conj(n_c[ikrdgamma]);
							oosum += o_c[ikgamma]*conj(o_c[ikrdgamma]);
							losum += l_c[ikgamma]*o_c[ikrdgamma];
							mnsum += m_c[ikgamma]*n_c[ikrdgamma];
							olsum += o_c[ikgamma]*l_c[ikrdgamma];
							nmsum += n_c[ikgamma]*m_c[ikrdgamma];
						}
					}
					
					// \alpha^dagger term l_ddt
					scterm1 = (l_c[ikrddelta]*llsum-conj(o_c[ikrddelta])*losum)*Rk+losum*l_c[ikrddelta]*Ck+llsum*conj(o_c[ikrddelta])*conj(Ck);
					scterm2 = (l_c[ikrddelta]*mmsum-conj(o_c[ikrddelta])*mnsum)*Rk+mnsum*l_c[ikrddelta]*Ck+mmsum*conj(o_c[ikrddelta])*conj(Ck);
					
					adagddt += scterm1 + scterm2;
					
					// \beta term m_ddt
					scterm1 = (m_c[ikrddelta]*mmsum-conj(n_c[ikrddelta])*mnsum)*Rk+mnsum*m_c[ikrddelta]*Ck+mmsum*conj(n_c[ikrddelta])*conj(Ck);
					scterm2 = (m_c[ikrddelta]*llsum-conj(n_c[ikrddelta])*losum)*Rk+losum*m_c[ikrddelta]*Ck+llsum*conj(n_c[ikrddelta])*conj(Ck);
					
					bddt += scterm1 + scterm2;
					
					// \alpha term l_ddt
					scterm1 = (conj(l_c[ikrddelta])*olsum-o_c[ikrddelta]*oosum)*Rk+olsum*o_c[ikrddelta]*Ck+oosum*conj(l_c[ikrddelta])*conj(Ck);
					scterm2 = (conj(l_c[ikrddelta])*nmsum-o_c[ikrddelta]*nnsum)*Rk+nmsum*o_c[ikrddelta]*Ck+nnsum*conj(l_c[ikrddelta])*conj(Ck);
					
					addt += scterm1 + scterm2;

					// \beta^dagger term m_ddt
					scterm1 = (conj(m_c[ikrddelta])*nmsum-n_c[ikrddelta]*nnsum)*Rk+nmsum*n_c[ikrddelta]*Ck+nnsum*conj(m_c[ikrddelta])*conj(Ck);
					scterm2 = (conj(m_c[ikrddelta])*olsum-n_c[ikrddelta]*oosum)*Rk+olsum*n_c[ikrddelta]*Ck+oosum*conj(m_c[ikrddelta])*conj(Ck);

					bdagddt += scterm1 + scterm2;
					
  					///////////////////////////////////
					
					//// q = 0
					
					// k'+\rho+\delta-q
					iLM3 = ikrd*3+1;
					// \alpha^dagger term l_ddt
					emterm1 = (l_c[ikrddelta]*llsum-conj(o_c[ikrddelta])*losum)*Lm[iLM3]-(losum*l_c[ikrddelta]+llsum*conj(o_c[ikrddelta]))*Mp[iLM3];
					emterm2 = (l_c[ikrddelta]*mmsum-conj(o_c[ikrddelta])*mnsum)*Lm[iLM3]-(mnsum*l_c[ikrddelta]+mmsum*conj(o_c[ikrddelta]))*Mp[iLM3];
					
					adagddt += Aqpm*(emterm1+emterm2);
					
					// \beta term m_ddt
					emterm1 = (m_c[ikrddelta]*mmsum-conj(n_c[ikrddelta])*mnsum)*Lm[iLM3]-(mnsum*m_c[ikrddelta]+mmsum*conj(n_c[ikrddelta]))*Mp[iLM3];
					emterm2 = (m_c[ikrddelta]*llsum-conj(n_c[ikrddelta])*losum)*Lm[iLM3]-(losum*m_c[ikrddelta]+llsum*conj(n_c[ikrddelta]))*Mp[iLM3];
					
					bddt += Aqpm*(emterm1+emterm2);
					
					// \alpha term l_ddt
					emterm1 = (conj(l_c[ikrddelta])*olsum-o_c[ikrddelta]*oosum)*Lm[iLM3]-(olsum*o_c[ikrddelta]+oosum*conj(l_c[ikrddelta]))*Mp[iLM3];
					emterm2 = (conj(l_c[ikrddelta])*nmsum-o_c[ikrddelta]*nnsum)*Lm[iLM3]-(nmsum*o_c[ikrddelta]+nnsum*conj(l_c[ikrddelta]))*Mp[iLM3];
					
					addt += Aqpm*(emterm1+emterm2);

					// \beta^dagger term m_ddt
					emterm1 = (conj(m_c[ikrddelta])*nmsum-n_c[ikrddelta]*nnsum)*Lm[iLM3]-(nmsum*n_c[ikrddelta]+nnsum*conj(m_c[ikrddelta]))*Mp[iLM3];
					emterm2 = (conj(m_c[ikrddelta])*olsum-n_c[ikrddelta]*oosum)*Lm[iLM3]-(olsum*n_c[ikrddelta]+oosum*conj(m_c[ikrddelta]))*Mp[iLM3];

					bdagddt += Aqpm*(emterm1+emterm2);
				
					
					///////////////////////////////////
						
					//// q = -2q0
					iq = -2;
					// EM terms 3&4
					i_krhodeltapq = i_krhodelta+iq;
					if(i_krhodeltapq>=0 && i_krhodeltapq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"
						
						// Reset Sums to zero for EM terms 3&4
						nnsum = 0.0;
						oosum = 0.0;
						olsum = 0.0;
						nmsum = 0.0;
						for (igamma = 0; igamma<Ndelta; igamma++){
							if (abs(irho-idelta-igamma+iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltapq,igamma+i_deltarho-iq);

								
								// Accumulate sums over gamma
								nnsum += n_c[ikgamma]*conj(n_c[ikgammaq]);
								oosum += o_c[ikgamma]*conj(o_c[ikgammaq]);
								olsum += o_c[ikgamma]*l_c[ikgammaq];
								nmsum += n_c[ikgamma]*m_c[ikgammaq];
							}
						}
						// k'+\rho-\eta
						iLM3 = ikrd*3;
						// \alpha term l_ddt
						emterm1 = (conj(l_c[ikrddelta])*olsum-o_c[ikrddelta]*oosum)*Lm[iLM3]-(olsum*o_c[ikrddelta]+oosum*conj(l_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (conj(l_c[ikrddelta])*nmsum-o_c[ikrddelta]*nnsum)*Lm[iLM3]-(nmsum*o_c[ikrddelta]+nnsum*conj(l_c[ikrddelta]))*Mp[iLM3];
						
						addt += Aqm2*(emterm1+emterm2);

						// \beta^dagger term m_ddt
						emterm1 = (conj(m_c[ikrddelta])*nmsum-n_c[ikrddelta]*nnsum)*Lm[iLM3]-(nmsum*n_c[ikrddelta]+nnsum*conj(m_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (conj(m_c[ikrddelta])*olsum-n_c[ikrddelta]*oosum)*Lm[iLM3]-(olsum*n_c[ikrddelta]+oosum*conj(m_c[ikrddelta]))*Mp[iLM3];

						bdagddt += Aqm2*(emterm1+emterm2);
					
					}
						
					// EM Terms 1&2
					i_krhodeltamq = i_krhodelta-iq;
					if(i_krhodeltamq>=0 && i_krhodeltamq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"

					
						// ReSet Sums to zero for EM terms 1&2
						llsum = 0.0;
						mmsum = 0.0;
						losum = 0.0;
						mnsum = 0.0;
							
						for (igamma = 0; igamma<Ndelta; igamma++){
							// EM TERMS 1&2 CONDITIONAL
							if (abs(irho-idelta-igamma-iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltamq,igamma+i_deltarho+iq);
															
								// Accumulate sums over gamma
								llsum += l_c[ikgamma]*conj(l_c[ikgammaq]);
								mmsum += m_c[ikgamma]*conj(m_c[ikgammaq]);
								losum += l_c[ikgamma]*o_c[ikgammaq];
								mnsum += m_c[ikgamma]*n_c[ikgammaq];
							}
						}
						
						
						
						// k'+\rho+\delta-q
						iLM3 = (ikrd-iq)*3;
						// \alpha^dagger term l_ddt
						emterm1 = (l_c[ikrddelta]*llsum-conj(o_c[ikrddelta])*losum)*Lm[iLM3]-(losum*l_c[ikrddelta]+llsum*conj(o_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (l_c[ikrddelta]*mmsum-conj(o_c[ikrddelta])*mnsum)*Lm[iLM3]-(mnsum*l_c[ikrddelta]+mmsum*conj(o_c[ikrddelta]))*Mp[iLM3];
						
						adagddt += Aqm2*(emterm1+emterm2);
						
						// \beta term m_ddt
						emterm1 = (m_c[ikrddelta]*mmsum-conj(n_c[ikrddelta])*mnsum)*Lm[iLM3]-(mnsum*m_c[ikrddelta]+mmsum*conj(n_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (m_c[ikrddelta]*llsum-conj(n_c[ikrddelta])*losum)*Lm[iLM3]-(losum*m_c[ikrddelta]+llsum*conj(n_c[ikrddelta]))*Mp[iLM3];
						
						bddt += Aqm2*(emterm1+emterm2);
					}
					
					/////////////////////////////////// ///////// 
						
					//// q = -q0
					iq = -1;
					// EM terms 3&4
					i_krhodeltapq = i_krhodelta+iq;
					if(i_krhodeltapq>=0 && i_krhodeltapq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"
						
						// Reset Sums to zero for EM terms 3&4
						nnsum = 0.0;
						oosum = 0.0;
						olsum = 0.0;
						nmsum = 0.0;
						for (igamma = 0; igamma<Ndelta; igamma++){
							if (abs(irho-idelta-igamma+iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltapq,igamma+i_deltarho-iq);

								
								// Accumulate sums over gamma
								nnsum += n_c[ikgamma]*conj(n_c[ikgammaq]);
								oosum += o_c[ikgamma]*conj(o_c[ikgammaq]);
								olsum += o_c[ikgamma]*l_c[ikgammaq];
								nmsum += n_c[ikgamma]*m_c[ikgammaq];
							}
						}
						iLM2 = ikrd*2; // k'+\rho-\eta
						// \alpha term l_ddt
						emterm1 = (conj(l_c[ikrddelta])*olsum+o_c[ikrddelta]*oosum)*Lp[iLM2]+(-olsum*o_c[ikrddelta]+oosum*conj(l_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (conj(l_c[ikrddelta])*nmsum+o_c[ikrddelta]*nnsum)*Lp[iLM2]+(-nmsum*o_c[ikrddelta]+nnsum*conj(l_c[ikrddelta]))*Mm[iLM2];
						
						addt += Aqm1*(emterm1+emterm2);

						// \beta^dagger term m_ddt
						emterm1 = (conj(m_c[ikrddelta])*nmsum+n_c[ikrddelta]*nnsum)*Lp[iLM2]+(-nmsum*n_c[ikrddelta]+nnsum*conj(m_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (conj(m_c[ikrddelta])*olsum+n_c[ikrddelta]*oosum)*Lp[iLM2]+(-olsum*n_c[ikrddelta]+oosum*conj(m_c[ikrddelta]))*Mm[iLM2];

						bdagddt += Aqm1*(emterm1+emterm2);
					
					}
						
					// EM Terms 1&2
					i_krhodeltamq = i_krhodelta-iq;
					if(i_krhodeltamq>=0 && i_krhodeltamq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"

					
						// ReSet Sums to zero for EM terms 1&2
						llsum = 0.0;
						mmsum = 0.0;
						losum = 0.0;
						mnsum = 0.0;
							
						for (igamma = 0; igamma<Ndelta; igamma++){
							// EM TERMS 1&2 CONDITIONAL
							if (abs(irho-idelta-igamma-iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltamq,igamma+i_deltarho+iq);

								// Accumulate sums over gamma
								llsum += l_c[ikgamma]*conj(l_c[ikgammaq]);
								mmsum += m_c[ikgamma]*conj(m_c[ikgammaq]);
								losum += l_c[ikgamma]*o_c[ikgammaq];
								mnsum += m_c[ikgamma]*n_c[ikgammaq];
							}
						}
						
						iLM2 = (ikrd-iq)*2; // k'+\rho+\delta-q
						// \alpha^dagger term l_ddt
						emterm1 = (l_c[ikrddelta]*llsum+conj(o_c[ikrddelta])*losum)*Lp[iLM2]+(-losum*l_c[ikrddelta]+llsum*conj(o_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (l_c[ikrddelta]*mmsum+conj(o_c[ikrddelta])*mnsum)*Lp[iLM2]+(-mnsum*l_c[ikrddelta]+mmsum*conj(o_c[ikrddelta]))*Mm[iLM2];
						
						adagddt += Aqm1*(emterm1+emterm2);
						
						// \beta term m_ddt
						emterm1 = (m_c[ikrddelta]*mmsum+conj(n_c[ikrddelta])*mnsum)*Lp[iLM2]+(-mnsum*m_c[ikrddelta]+mmsum*conj(n_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (m_c[ikrddelta]*llsum+conj(n_c[ikrddelta])*losum)*Lp[iLM2]+(-losum*m_c[ikrddelta]+llsum*conj(n_c[ikrddelta]))*Mm[iLM2];
						
						bddt += Aqm1*(emterm1+emterm2);
					}				
						
					///////////////////////////////////
						
					//// q = +q0
					iq = 1;
					// EM terms 3&4
					i_krhodeltapq = i_krhodelta+iq;
					if(i_krhodeltapq>=0 && i_krhodeltapq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"
						
						// Reset Sums to zero for EM terms 3&4
						nnsum = 0.0;
						oosum = 0.0;
						olsum = 0.0;
						nmsum = 0.0;
						for (igamma = 0; igamma<Ndelta; igamma++){
							if (abs(irho-idelta-igamma+iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltapq,igamma+i_deltarho-iq);

								
								// Accumulate sums over gamma
								nnsum += n_c[ikgamma]*conj(n_c[ikgammaq]);
								oosum += o_c[ikgamma]*conj(o_c[ikgammaq]);
								olsum += o_c[ikgamma]*l_c[ikgammaq];
								nmsum += n_c[ikgamma]*m_c[ikgammaq];
							}
						}
						iLM2 = ikrd*2+1; // k'+\rho-\eta
						// \alpha term l_ddt
						emterm1 = (conj(l_c[ikrddelta])*olsum+o_c[ikrddelta]*oosum)*Lp[iLM2]+(-olsum*o_c[ikrddelta]+oosum*conj(l_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (conj(l_c[ikrddelta])*nmsum+o_c[ikrddelta]*nnsum)*Lp[iLM2]+(-nmsum*o_c[ikrddelta]+nnsum*conj(l_c[ikrddelta]))*Mm[iLM2];
						
						addt += Aqp1*(emterm1+emterm2);

						// \beta^dagger term m_ddt
						emterm1 = (conj(m_c[ikrddelta])*nmsum+n_c[ikrddelta]*nnsum)*Lp[iLM2]+(-nmsum*n_c[ikrddelta]+nnsum*conj(m_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (conj(m_c[ikrddelta])*olsum+n_c[ikrddelta]*oosum)*Lp[iLM2]+(-olsum*n_c[ikrddelta]+oosum*conj(m_c[ikrddelta]))*Mm[iLM2];

						bdagddt += Aqp1*(emterm1+emterm2);
					
					}
						
					// EM Terms 1&2
					i_krhodeltamq = i_krhodelta-iq;
					if(i_krhodeltamq>=0 && i_krhodeltamq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"

					
						// ReSet Sums to zero for EM terms 1&2
						llsum = 0.0;
						mmsum = 0.0;
						losum = 0.0;
						mnsum = 0.0;
							
						for (igamma = 0; igamma<Ndelta; igamma++){
							// EM TERMS 1&2 CONDITIONAL
							if (abs(irho-idelta-igamma-iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltamq,igamma+i_deltarho+iq);

								// Accumulate sums over gamma
								llsum += l_c[ikgamma]*conj(l_c[ikgammaq]);
								mmsum += m_c[ikgamma]*conj(m_c[ikgammaq]);
								losum += l_c[ikgamma]*o_c[ikgammaq];
								mnsum += m_c[ikgamma]*n_c[ikgammaq];
							}
						}
						
						iLM2 = (ikrd-iq)*2+1; // k'+\rho+\delta-q
						// \alpha^dagger term l_ddt
						emterm1 = (l_c[ikrddelta]*llsum+conj(o_c[ikrddelta])*losum)*Lp[iLM2]+(-losum*l_c[ikrddelta]+llsum*conj(o_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (l_c[ikrddelta]*mmsum+conj(o_c[ikrddelta])*mnsum)*Lp[iLM2]+(-mnsum*l_c[ikrddelta]+mmsum*conj(o_c[ikrddelta]))*Mm[iLM2];
						
						adagddt += Aqp1*(emterm1+emterm2);
						
						// \beta term m_ddt
						emterm1 = (m_c[ikrddelta]*mmsum+conj(n_c[ikrddelta])*mnsum)*Lp[iLM2]+(-mnsum*m_c[ikrddelta]+mmsum*conj(n_c[ikrddelta]))*Mm[iLM2];
						emterm2 = (m_c[ikrddelta]*llsum+conj(n_c[ikrddelta])*losum)*Lp[iLM2]+(-losum*m_c[ikrddelta]+llsum*conj(n_c[ikrddelta]))*Mm[iLM2];
						
						bddt += Aqp1*(emterm1+emterm2);
					}					

					///////////////////////////////////
						
					//// q = +2q0
					iq = 2;
					// EM terms 3&4
					i_krhodeltapq = i_krhodelta+iq;
					if(i_krhodeltapq>=0 && i_krhodeltapq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"
						
						// Reset Sums to zero for EM terms 3&4
						nnsum = 0.0;
						oosum = 0.0;
						olsum = 0.0;
						nmsum = 0.0;
						for (igamma = 0; igamma<Ndelta; igamma++){
							if (abs(irho-idelta-igamma+iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltapq,igamma+i_deltarho-iq);

								
								// Accumulate sums over gamma
								nnsum += n_c[ikgamma]*conj(n_c[ikgammaq]);
								oosum += o_c[ikgamma]*conj(o_c[ikgammaq]);
								olsum += o_c[ikgamma]*l_c[ikgammaq];
								nmsum += n_c[ikgamma]*m_c[ikgammaq];
							}
						}
						// k'+\rho-\eta
						iLM3 = ikrd*3+2;
						// \alpha term l_ddt
						emterm1 = (conj(l_c[ikrddelta])*olsum-o_c[ikrddelta]*oosum)*Lm[iLM3]-(olsum*o_c[ikrddelta]+oosum*conj(l_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (conj(l_c[ikrddelta])*nmsum-o_c[ikrddelta]*nnsum)*Lm[iLM3]-(nmsum*o_c[ikrddelta]+nnsum*conj(l_c[ikrddelta]))*Mp[iLM3];
						
						addt += Aqp2*(emterm1+emterm2);

						// \beta^dagger term m_ddt
						emterm1 = (conj(m_c[ikrddelta])*nmsum-n_c[ikrddelta]*nnsum)*Lm[iLM3]-(nmsum*n_c[ikrddelta]+nnsum*conj(m_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (conj(m_c[ikrddelta])*olsum-n_c[ikrddelta]*oosum)*Lm[iLM3]-(olsum*n_c[ikrddelta]+oosum*conj(m_c[ikrddelta]))*Mp[iLM3];

						bdagddt += Aqp2*(emterm1+emterm2);
					
					}
						
					// EM Terms 1&2
					i_krhodeltamq = i_krhodelta-iq;
					if(i_krhodeltamq>=0 && i_krhodeltamq<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"

					
						// ReSet Sums to zero for EM terms 1&2
						llsum = 0.0;
						mmsum = 0.0;
						losum = 0.0;
						mnsum = 0.0;
							
						for (igamma = 0; igamma<Ndelta; igamma++){
							// EM TERMS 1&2 CONDITIONAL
							if (abs(irho-idelta-igamma-iq+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
								ikgamma = index(itheta, ik, igamma); // index at k'+\gamma
								ikgammaq = index(itheta, i_krhodeltamq,igamma+i_deltarho+iq);

								// Accumulate sums over gamma
								llsum += l_c[ikgamma]*conj(l_c[ikgammaq]);
								mmsum += m_c[ikgamma]*conj(m_c[ikgammaq]);
								losum += l_c[ikgamma]*o_c[ikgammaq];
								mnsum += m_c[ikgamma]*n_c[ikgammaq];
							}
						}
						
						// k'+\rho+\delta-q
						iLM3 = (ikrd-iq)*3+2;
						// \alpha^dagger term l_ddt
						emterm1 = (l_c[ikrddelta]*llsum-conj(o_c[ikrddelta])*losum)*Lm[iLM3]-(losum*l_c[ikrddelta]+llsum*conj(o_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (l_c[ikrddelta]*mmsum-conj(o_c[ikrddelta])*mnsum)*Lm[iLM3]-(mnsum*l_c[ikrddelta]+mmsum*conj(o_c[ikrddelta]))*Mp[iLM3];
						
						adagddt += Aqp2*(emterm1+emterm2);
						
						// \beta term m_ddt
						emterm1 = (m_c[ikrddelta]*mmsum-conj(n_c[ikrddelta])*mnsum)*Lm[iLM3]-(mnsum*m_c[ikrddelta]+mmsum*conj(n_c[ikrddelta]))*Mp[iLM3];
						emterm2 = (m_c[ikrddelta]*llsum-conj(n_c[ikrddelta])*losum)*Lm[iLM3]-(losum*m_c[ikrddelta]+llsum*conj(n_c[ikrddelta]))*Mp[iLM3];
						
						bddt += Aqp2*(emterm1+emterm2);
					}
				}				
			}
						
			l_ddt[ikrho]  = I/hbar*adagddt; // negative I cancelled					
			m_ddt[ikrho]  = I/hbar*bddt; // negative I cancelled
			o_ddt[ikrho]  = -I/hbar*addt; 
			n_ddt[ikrho]  = -I/hbar*bdagddt;			
			
			/* if (indicator==0 && (real(adagddt)!=0.0 || real(bdagddt)!=0.0 || real(addt)!=0.0 || real(bddt)!=0.0)){
				printf("iTheta %d\t%e\t%e\t%e\t%e\t%e\n",itheta,t,real(adagddt), real(bdagddt), real(addt), real(bddt));
				indicator =1;
			} */
			

		}
	}
	//printf("iTheta %d\t%e\t%e\t%e\t%e\t%e\n",itheta,t,Mp[iLM3],Mm[iLM2],Lp[iLM2],Lm[iLM3]);
	//printf("iTheta %d\t%e\t%e\t%e\t%e\t%e\n",itheta,t,real(adagddt), real(bdagddt), real(addt), real(bddt));


}

//////////////////////////////////////
///// For While the Pump is Off //////
//////////////////////////////////////
void diffEq2(int itheta, double t, dcomp Delta_c, dcomp l_c[], dcomp m_c[], dcomp n_c[], dcomp o_c[]){
	
	// variables for various indices 
    int idelta, i, igamma, ikrd, irho, ik, ikrho, ikgamma, ikrdgamma, ikrddelta, i_krhodelta, i_krho, i_deltarho;
	
	    
	// To accumulate the sums before
	double Rk;
	dcomp Ck, Deltak;
	dcomp losum, mnsum, olsum, nmsum;
    dcomp llsum, mmsum, nnsum, oosum;
	
	// Helpful  terms for breaking eq up
	dcomp scterm1, scterm2;
	
	dcomp adagddt, addt, bdagddt, bddt;
	
	// Beginning  of the loop through  Nk x N_\rho
    for (ik = 0; ik<Nk[itheta]; ik++){
		for (irho = 0; irho<Ndelta; irho++){
			ikrho = index(itheta, ik, irho);
			i_krho = ik+irho;
			// Set sums to zero
			
			adagddt = 0.0;
			addt = 0.0;
			bdagddt = 0.0;
			bddt = 0.0;
		
			
			// Accumulate all the sums  over \delta, \eta
			for (idelta = 0; idelta<Ndelta; idelta++){ // sum over \delta (replace \eta by \delta for notation equivalency)
				
				i_deltarho = idelta-irho; // \delta-\rho
					
				// Foor indexing at ikrd = k'+\rho-\delta 
				i_krhodelta = i_krho-idelta; // first term used in indexing l,m,n,o
				
				// index for k' + rho - delta in R,C,L,M
				ikrd = indexRC(itheta, i_krhodelta+2*Ndhalf); // i = k ' + \rho - \delta;
				
				ikrddelta = index(itheta, i_krhodelta, idelta); // index for k'+\rho-\delta,\delta for l,m,n,o

				if(i_krhodelta>=0 && i_krhodelta<Nk[itheta]){ // check if k'+\rho-\delta is  in our "W"
					
					
					// Delta at current t
					Deltak=Wk[ikrd]*Delta_c; 
										
					Rk = R1[ikrd]+R2[ikrd]*real(Deltak);
					Ck = C1[ikrd]+C2[ikrd]*Deltak-C3[ikrd]*conj(Deltak);
										
					// Set Sums to zero
					llsum = 0.0;
					mmsum = 0.0;
					nnsum = 0.0;
					oosum = 0.0;
					losum = 0.0;
					mnsum = 0.0;
					olsum = 0.0;
					nmsum = 0.0;
					
					for(igamma = 0; igamma<Ndelta; igamma++){ // most inner  sum over \gamma
						// SC TERM CONDITIONAL
						if (abs(irho-idelta-igamma+Ndhalf)<(Ndhalf+1)){ // conditional for Ndelta cutoff 
							ikgamma = index(itheta, ik, igamma); // index at k',\gamma
							ikrdgamma = index(itheta,i_krhodelta,igamma+i_deltarho); // index for k'+\rho-\delta,\gamma+\delta-\rho
							
							
							// Accumulate sums over gamma
							llsum += l_c[ikgamma]*conj(l_c[ikrdgamma]);
							mmsum += m_c[ikgamma]*conj(m_c[ikrdgamma]);
							nnsum += n_c[ikgamma]*conj(n_c[ikrdgamma]);
							oosum += o_c[ikgamma]*conj(o_c[ikrdgamma]);
							losum += l_c[ikgamma]*o_c[ikrdgamma];
							mnsum += m_c[ikgamma]*n_c[ikrdgamma];
							olsum += o_c[ikgamma]*l_c[ikrdgamma];
							nmsum += n_c[ikgamma]*m_c[ikrdgamma];
						}
					}
					
					// \alpha^dagger term l_ddt
					scterm1 = (l_c[ikrddelta]*llsum-conj(o_c[ikrddelta])*losum)*Rk+losum*l_c[ikrddelta]*Ck+llsum*conj(o_c[ikrddelta])*conj(Ck);
					scterm2 = (l_c[ikrddelta]*mmsum-conj(o_c[ikrddelta])*mnsum)*Rk+mnsum*l_c[ikrddelta]*Ck+mmsum*conj(o_c[ikrddelta])*conj(Ck);
					
					adagddt += scterm1 + scterm2;
					
					// \beta term m_ddt
					scterm1 = (m_c[ikrddelta]*mmsum-conj(n_c[ikrddelta])*mnsum)*Rk+mnsum*m_c[ikrddelta]*Ck+mmsum*conj(n_c[ikrddelta])*conj(Ck);
					scterm2 = (m_c[ikrddelta]*llsum-conj(n_c[ikrddelta])*losum)*Rk+losum*m_c[ikrddelta]*Ck+llsum*conj(n_c[ikrddelta])*conj(Ck);
					
					bddt += scterm1 + scterm2;
					
					// \alpha term l_ddt
					scterm1 = (conj(l_c[ikrddelta])*olsum-o_c[ikrddelta]*oosum)*Rk+olsum*o_c[ikrddelta]*Ck+oosum*conj(l_c[ikrddelta])*conj(Ck);
					scterm2 = (conj(l_c[ikrddelta])*nmsum-o_c[ikrddelta]*nnsum)*Rk+nmsum*o_c[ikrddelta]*Ck+nnsum*conj(l_c[ikrddelta])*conj(Ck);
					
					addt += scterm1 + scterm2;

					// \beta^dagger term m_ddt
					scterm1 = (conj(m_c[ikrddelta])*nmsum-n_c[ikrddelta]*nnsum)*Rk+nmsum*n_c[ikrddelta]*Ck+nnsum*conj(m_c[ikrddelta])*conj(Ck);
					scterm2 = (conj(m_c[ikrddelta])*olsum-n_c[ikrddelta]*oosum)*Rk+olsum*n_c[ikrddelta]*Ck+oosum*conj(m_c[ikrddelta])*conj(Ck);

					bdagddt += scterm1 + scterm2;
				}
			}
			
			l_ddt[ikrho]  = I/hbar*adagddt; // negative I cancelled					
			m_ddt[ikrho]  = I/hbar*bddt; // negative I cancelled
			o_ddt[ikrho]  = -I/hbar*addt; 
			n_ddt[ikrho]  = -I/hbar*bdagddt;				

		}
	}
}

//////////////////////////////////////////////////////////////
/////		Delta Calculation & W Initiliazation /////////////
///// 			_c for at the current time       /////////////
//////////////////////////////////////////////////////////////
dcomp gapEq(dcomp m_c[], dcomp n_c[], dcomp o_c[]){ 
	double sumDeltaR = 0.0;
	double sumDeltaI = 0.0;
	int itheta;
#pragma omp parallel for schedule(dynamic,1) reduction(+:sumDeltaR,sumDeltaI)
	for (itheta=0; itheta<Ntheta; itheta++){
		int i, ik, idelta;
		dcomp sumDelta2 = 0.0;
		for (ik=0; ik<Nk[itheta]; ik++){
			double msum = 0.0;
			double osum = 0.0;
		
			dcomp nmsum = 0.0;
			for (idelta=0; idelta<Ndelta; idelta++){
				i = index(itheta,ik,idelta); // index for l,m,n,o
				msum += norm(m_c[i]);
				osum += norm(o_c[i]);
				
				nmsum += n_c[i]*m_c[i];
				/* if (indicator==0 && (real(nmsum)!=0.0 || (osum)!=0.0 || (msum)!=0.0 || imag(nmsum)!=0.0)){
					printf("iTheta %d\t%e\t%d\t%d\n",itheta,t,ik,idelta);
					indicator =1;
				} */
			}
			i = indexRC(itheta,ik+Ndhalf*2); // index for only k in W, Nk.
			sumDelta2 += Wk[i]*(R2[i]/2.0*dcomp(msum+osum-1.0)+C2[i]*conj(nmsum)-C3[i]*nmsum);
		}
		sumDeltaR += real(sumDelta2/dcomp(Nk[itheta]));
		sumDeltaI += imag(sumDelta2/dcomp(Nk[itheta]));
	}
	
	return W*dcomp(sumDeltaR,sumDeltaI)/dcomp(Ntheta);
}

void calcW(){ // In progress for indices!!!!!!!!!!!!!!!
	int itheta;
	double sW = 0.0;
#pragma omp parallel for schedule(dynamic,1) reduction(+: sW)
	for (itheta = 0; itheta<Ntheta; itheta++){
		int ik, i;
		double sumW2 = 0.0;
		for(ik=0; ik<Nk[itheta]; ik++){
			i = indexRC(itheta,ik+Ndhalf*2);
			sumW2 += Wk[i]*R2[i];
		}
		sW += sumW2/double(Nk[itheta])/2.0; // R2 is twice as big
	}
	W = -Delta0/sW*Ntheta;
}

/////////////////////////////////////////////////////////////
////		Green's Function Calculation         ////////////
/////////////////////////////////////////////////////////////
void printGreensVariables(){
	int i, idelta, myIndex;
	FILE *greensOut;
	greensOut = fopen("lmno.txt","a");
	for (i = 0; i<kFpoints; i++){
		myIndex = idxkF[i];
		for (idelta = 0; idelta<Ndelta; idelta++){
			fprintf(greensOut,"%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",t,thetakF[i],real(l[myIndex]),imag(l[myIndex]),real(m[myIndex]),imag(m[myIndex]),real(n[myIndex]),imag(n[myIndex]),real(o[myIndex]),imag(o[myIndex]));
			myIndex++;
		}
	}
	fclose(greensOut);
}

//////////////////////////////////////////////////////////////
//         Angle Functions (on the FS)                    ////
//	Initilizes the momentum grid, W, pre-calculations     ////
//////////////////////////////////////////////////////////////
void calcAngles(){ // must be called before any call to any FS variables
    int itheta, i, ik, irr;
    double thetap, theta, r1, r2, l1, l2, kx, ky, k;
	printf("Into the Array Calculation Function\n");
       
    r1 = sqrt((Ef-hbar_wc)/t0); // inner radius
    r2 = sqrt((Ef+hbar_wc)/t0); // outter radius
    
    thetac = asin(r1/r2); // Theta-cutoff near the top
	i = 0;
    
    // Calculate the K-points for each Theta
    for (itheta = 0; itheta<Ntheta; itheta++){
        theta = intToAngle(itheta);
        
        if (theta < thetac || (((pi-thetac) < theta) && ((pi+thetac) > theta)) ||
                (2*pi-thetac)<theta){ // The  Regular Section!
            
            thetap = asin(r2*sin(theta)/r1)-theta;
			
			i++;
            
            ky = r2*sin(theta);
            kx = r1*cos(theta+thetap);
            
            l1 = abs(kx);
            l2 = abs(r2*cos(theta));
            
            Nk[itheta] = round((l2-l1)/q0)+1;
            Nkrho[itheta] = Nk[itheta]+2*Ndhalf;
			Nkrd[itheta] = Nk[itheta]+4*Ndhalf;
            
            kFy[itheta] = ky;
            if (theta > pi/2.0 && theta < 3.0*pi/2.0)
                kx0[itheta] = r2*cos(theta);
            else 

                kx0[itheta] = kx;
        }
        else{ // The Irregular Section!
			
			
            ky = r2*sin(theta);
            kx = r2*cos(theta);
            
			if(abs(ky)<kF)
				i += 2;
			
            l1 = abs(kx);
            
            Nk[itheta] = round(2.0*l1/q0)+1;
            Nkrho[itheta] = Nk[itheta]+2*Ndhalf;           
			Nkrd[itheta] = Nk[itheta]+4*Ndhalf;
			
            kFy[itheta] = ky;
            kx0[itheta] = -abs(kx);
        }
    }
	idxkF = new int[i];
	thetakF = new double[i];
	kFpoints = i;

	
    
    int idx[Ntheta];
    size_t sizeI = sizeof(idx) / sizeof(idx[0]);  
    for (itheta = 0; itheta<Ntheta; itheta++){
        idx[itheta] = itheta;
    }
    sort(idx,idx+sizeI,kSort);
    
    arrayIorder(Nkrho, idx);
    arrayIorder(Nk, idx);
	arrayIorder(Nkrd,idx);
    arrayDorder(kFy, idx);
    arrayDorder(kx0, idx);
        
    // must be called after array re-ordering!
    arrayLengths();
	
	i = 0;
	
	for (itheta = 0; itheta<Ntheta; itheta++){
        theta = intToAngle(idx[itheta]);
		ky = kFy[itheta];
        
        if (theta < thetac || (((pi-thetac) < theta) && ((pi+thetac) > theta)) || (2*pi-thetac)<theta){ // The  Regular Section!
			for(ik = 0; ik<Nk[itheta]; ik++){
				kx = double(ik)*q0+kx0[itheta];
				k = sqrt(pow(kx,2.0)+pow(ky,2.0));
				if(abs(k-kF)<=q0){
					idxkF[i] = index(itheta,ik,0);
					thetakF[i] = atan2(ky,kx);
					i++;
					break;
				}
				
			}
		}
		else if(abs(ky)<kF){ // The Irregular Section!
			irr = 0;
			for(ik = 0; ik<Nk[itheta]; ik++){
				if (irr>1)
					break;
				kx = double(ik)*q0+kx0[itheta];
				k = sqrt(pow(kx,2.0)+pow(ky,2.0));
				if(abs(k-kF)<=q0){
					idxkF[i] = index(itheta,ik,0);
					thetakF[i] = atan2(ky,kx);
					i++;
					ik++;
					irr++;
				}
			}
		}
	}
    
	printf("Printing Kpoints File\n");
    // Print to file
    FILE *Kpoints;
    Kpoints = fopen("Kpoints.txt","w");
    fprintf(Kpoints, "Theta \t");
    fprintf(Kpoints, "Kpoints \t");
    fprintf(Kpoints, "kx0 \t");
    fprintf(Kpoints, "ky \n");    
    for (itheta = 0; itheta<Ntheta; itheta++){
        
        kdotA[itheta] = 2.0*kFy[itheta]*A0eff1;
        //kdotA_pr[itheta] = 2.0*kFy[itheta]*A0eff1_pr;
        
        // Define spot on Fermi Surface
        double theta = intToAngle(idx[itheta]);
        
        fprintf(Kpoints, "%f\t", theta);
        fprintf(Kpoints, "%d\t", Nkrd[itheta]);
        fprintf(Kpoints, "%f\t", kx0[itheta]);
        fprintf(Kpoints, "%f\n", kFy[itheta]);
        
    }
	fprintf(Kpoints, "\nkF points %d\n",kFpoints);
	for (i = 0; i<kFpoints; i++){
		fprintf(Kpoints, "%f\t%d\n",idxkF[i],thetakF[i]);
	}
    fclose(Kpoints);

}

///////////////////////////////////////////////////////////
/////		Function for Pre-calculating Varibles	///////
///////////////////////////////////////////////////////////
void preCalcs(){
	int itheta;		
    for (itheta = 0; itheta<Ntheta; itheta++){
		int i, ik, iLM2, iLM3, iRC;
		
		double epsk, Ek, uk, vk, epskq, Ekq, ukq, vkq;
		
		double kx, ksq, ktheta, kqx, kqsq, kqtheta, Wkq, Deltakq0;
		
		for (ik = 0; ik<Nkrd[itheta]; ik++){
			i = indexRC(itheta, ik); // i = k'+rho-delta array index
			kx = double(ik)*q0+kx0[itheta];
		
			// Delta_k calculations
			ktheta = atan((kFy[itheta])/kx);
			Wk[i] = dwave(ktheta);
			Deltak0[i]=Wk[i]*Delta0; /// Pre-calculated
		}
		
		// Got to Here!!

		
		for (ik = 0; ik<Nkrd[itheta]; ik++){
    		i = indexRC(itheta, ik); // i = k'+rho-delta array index
			kx = double(ik)*q0+kx0[itheta];
		
			ksq = kx*kx+kFy[itheta]*kFy[itheta];
		
			// Energy Calculations
			epsk = t0*ksq-Ef;
			Ek= sqrt(pow(epsk,2.0)+pow(Deltak0[i],2.0));

			// u_k and v_k calculations
			uk=sqrt(0.5*(1.0+epsk/Ek));
			vk=copysign(sqrt(0.5*(1.0-epsk/Ek)), Wk[i]);
			
			// Pre-calculated terms in R and Delta
			R1[i] = epsk*(1.0-2.0*vk*vk);
			R2[i] = 2.0*uk*vk;
			
			// Here is the Problem Somehow!
			
			// THe prblem arrises from the Irregular theta points!!! THe lots of N values!
			
			// Pre-calculated terms  in C and Delta
			C1[i] = -2.0*epsk*uk*vk;
			C2[i] = uk*uk;
			C3[i] = vk*vk;
			
			iLM2 = i*2;
			iLM3 = i*3;
			// q = -q0 calculation:
			kqx = double(ik-1)*q0+kx0[itheta];
			kqsq = kqx*kqx+kFy[itheta]*kFy[itheta];
			epskq = t0*kqsq-Ef;
			kqtheta = atan((kFy[itheta])/kqx);
			Wkq = dwave(kqtheta);
			Deltakq0 = Wkq*Delta0;
			Ekq = sqrt(pow(epskq,2.0)+pow(Deltakq0,2.0));
			ukq = sqrt(0.5*(1.0+epskq/Ekq));
			vkq = copysign(sqrt(0.5*(1.0-epskq/Ekq)),Wkq);
			
			Lp[iLM2]=ukq*uk+vkq*vk;
			Mm[iLM2]=ukq*vk-vkq*uk;
			
			// q = +q0 calculation:
			kqx = double(ik+1)*q0+kx0[itheta];
			kqsq = kqx*kqx+kFy[itheta]*kFy[itheta];
			epskq = t0*kqsq-Ef;
			kqtheta = atan((kFy[itheta])/kqx);
			Wkq = dwave(kqtheta);
			Deltakq0 = Wkq*Delta0;
			Ekq = sqrt(pow(epskq,2.0)+pow(Deltakq0,2.0));
			ukq = sqrt(0.5*(1.0+epskq/Ekq));
			vkq = copysign(sqrt(0.5*(1.0-epskq/Ekq)),Wkq);

			Lp[iLM2+1]=ukq*uk+vkq*vk;
			Mm[iLM2+1]=ukq*vk-vkq*uk;
			
			// q = -2q0 calculation:
			kqx = double(ik-2)*q0+kx0[itheta];
			kqsq = kqx*kqx+kFy[itheta]*kFy[itheta];
			epskq = t0*kqsq-Ef;
			kqtheta = atan((kFy[itheta])/kqx);
			Wkq = dwave(kqtheta);
			Deltakq0 = Wkq*Delta0;
			Ekq = sqrt(pow(epskq,2.0)+pow(Deltakq0,2.0));
			ukq = sqrt(0.5*(1.0+epskq/Ekq));
			vkq = copysign(sqrt(0.5*(1.0-epskq/Ekq)),Wkq);
			
			Lm[iLM3]=ukq*uk-vkq*vk;
			Mp[iLM3]=vkq*uk+ukq*vk;
			
			// q = 0 calculation:
			Lm[iLM3+1]=C2[i]-C3[i];
			Mp[iLM3+1]=R2[i];
        
			// q= +2q0 calculation:
			kqx = double(ik+2)*q0+kx0[itheta];
			kqsq = kqx*kqx+kFy[itheta]*kFy[itheta];
			epskq = t0*kqsq-Ef;
			kqtheta = atan((kFy[itheta])/kqx);
			Wkq = dwave(kqtheta);
			Deltakq0 = Wkq*Delta0;
			Ekq = sqrt(pow(epskq,2.0)+pow(Deltakq0,2.0));
			ukq = sqrt(0.5*(1.0+epskq/Ekq));
			vkq = copysign(sqrt(0.5*(1.0-epskq/Ekq)),Wkq);
			
			Lm[iLM3+2]=ukq*uk-vkq*vk;
			Mp[iLM3+2]=vkq*uk+ukq*vk;
		}
		
	}
}


///////////////////////////////////////////////////////////
////////////// Functions for Sorting Theta points /////////
///////////////////////////////////////////////////////////
bool kSort(int i, int j){
    return Nk[i]>Nk[j];
}

void arrayIorder(int a[],  int idx[]){
    int b[Ntheta];
    for (int i=0; i<Ntheta; i++){
        b[i] = a[idx[i]];
    }
    for (int i = 0; i<Ntheta; i++){
        a[i] = b[i];
    }
}

void arrayDorder(double a[],  int idx[]){
    double b[Ntheta];
    for (int i=0; i<Ntheta; i++){
        b[i] = a[idx[i]];
    }
    for (int i = 0; i<Ntheta; i++){
        a[i] = b[i];
    }
}

//////////////////////////////////////////////////////////
//// Function to Accumulate Array Lengths ////////////////
// Calculate the length of the array preceding    ////////
//     each itheta section (& Nk section)         ////////
//////////////////////////////////////////////////////////
void arrayLengths(){
    int itheta;
    Lthetak[0] = 0;
    Ltkd[0] = 0;
    
    for (itheta = 1; itheta<Ntheta; itheta++){
        Lthetak[itheta] = Lthetak[itheta-1] + Nkrd[itheta-1];
        Ltkd[itheta] = Ltkd[itheta-1]+Nk[itheta-1]*Ndelta;
        
    }
}

///////////////////////////////////////////////
// Index Functions for Ntheta x Nk+4 //////////
//       & Ntheta x Nk x Ndelta      //////////
///////////////////////////////////////////////

int index(int itheta, int ik, int idelta){
	return Ltkd[itheta]+ik*Ndelta+idelta;
}

int indexRC(int itheta, int ik){
	return Lthetak[itheta]+ik;
}

int indexLM(int itheta, int ik, int Nq, int iq){
	return (Lthetak[itheta]+ik)*Nq+iq; // Not In Use
}


////////////////////////////////////////////////
////// Initialize angles ///////////////////////
// Warning! This does not indicate the angle ///
// as the indices are re-ordered!///////////////
////////////////////////////////////////////////
double intToAngle(int itheta){
    double theta, thetainterval;
	
    thetainterval =  (2.0*(pi/2.0+thetac))/Ntheta;
    theta = thetainterval*double(itheta);
    
/*     if (theta<pi/2.0)
        return theta;
    
    else if (theta+(pi/2.0-thetac)<3.0*pi/2.0){
        theta = theta+(pi/2.0-thetac);
        return theta;
    }
    else{
        theta = theta+2.0*(pi/2.0-thetac);
        return theta;
    }  */
    return atan(1.12/(sqrt((Ef/t0)-pow(1.12,2.0))));
}

//////////////////////////////
///////// Gap Symmetry ///////
//////////////////////////////
double dwave(double theta){
	return 1.0; //(swave)
    //return cos(2.0*theta+thetaFS); //(dwave)
}

