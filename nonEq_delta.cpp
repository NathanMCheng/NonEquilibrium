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
const double c = 2.99792458e-7;                                           // m / fs                speed of light (in units of the lattice constant)
const double me = 5.68563e21;                                              // meV fs^2 / m^2  electron mass

// Properties of the Superconductor
const double a = 1.0e-10;//3.82e-10;                                                 // m                     lattice constant
const double m = 1.9;//5.0;                                                      // 1/me                effecive mass (electron masses)
const double t0  = 3809.98174234/m; //pow(hbar,2.0)/2/m/me;                          // meV                 prefactor (2005.254 meV)
const double Ef = 9470.0;                                         // meV                 Fermi energy
const double Delta0 = 1.35;//20.0;                                              // meV                 gap energy (time-independent)
//const double ky = 0.0; //1.12;                                                    // a                       1/2 kF should be ?? (k_y momentum used)
//const double kF_x = sqrt((Ef/t0)-pow(ky,2.0));                             // a                       Fermi wavevector (2.173153)
const double kF = sqrt(Ef/t0);
const double hbar_wc = 8.3;                                            // meV                  cutoff energy (must be <1/2 Ef
//double W = -1773.0;                                               // meV                  interaction strength
//const double kU = sqrt((Ef+hbar_wD)/t0);                          // a                        upper wave vector limit kx
//const double kL = sqrt((Ef-hbar_wD)/t0);                            // a                        lower wave vector limit kx

// Pump Parameters
const double hbar_w0 = 3.0;           // meV                  pump energy
const double w0_p = hbar_w0/hbar;      // 1 / fs                  pump frequency (hbar_w0/hbar)
const double q0 = w0_p/c*a;        // a                       pump momentum transfer (q0 = w0/c)
const double tau_p = 400.0;        // fs                      pump full width at half maximum
const double A0 = 7.0e-8;          // (1e-8) J s / C m  pump intensity
const double A0eff1 = hbar/2.0/m/me/a*A0*1.0e18; //	meV effective pump intensity (1) 		  e hbar A0 / (2m a)
const double A0eff2 = 1.0/2.0/m/me*pow(A0,2.0)*1.0e36; // meV effective pump intensity (2)    e^2 A0^2 / (2m)
const double eh_2ma = hbar/2.0/m/me/a;

// Probe Parameters
const double hbar_w0_pr = 2.5;
const double w0_pr = hbar_w0_pr/hbar;
const double q_pr = w0_pr/c*a;
const double tau_pr = 250.0;
const double A0_pr = 1.0e-8;
const double A0eff1_pr = hbar/2.0/m/me/a*A0_pr*1.0e18;

// Define Momentum Array Variables
#define n 2200 //1462                                                                  // number of k elements ****** 2 times kbelow;
#define arrayBuffer 2                                                         // buffer to simplify calculation a boundaries
//#define N (n + 2*arrayBuffer)                                                // add buffer to arrays to prevent segmentation fault at boundaries
int Nd = 4;                                                                       //                          number of diagonal entries

#define Ndiag 13
#define Ntheta1 2000 
#define Ndt 1

const double thetaFS = pi/2.0;

const double h = 1.0;                                                               // fs
double tend = 700.0; // make sure to change dtmin if only looking at pump
const double dtmin = 300000.0;//250.0; (tend-dt_pr)
const double hdt = 100.0;


//double Deltak0[Ntheta1];

// FS Global Variables
double kx0[Ntheta1];
double kFy[Ntheta1];
double kdotA[Ntheta1];
double kdotA_pr[Ntheta1];
int Nk[Ntheta1];
int N[Ntheta1];
int N_pr;
int N_diag;

int Ltheta[Ntheta1];
int L_pr[Ntheta1];
int L_diag[Ntheta1];
double W; //[Ntheta1];

// SC Functions
double epsK(int ki, int theta1);             
double epsK_pr(int ki, int itheta1);                                           // meV                     band dispersion energy
double Ek(double eps_k, double Deltak);                                              // meV                    SC quasiparticle energy

// Bogoliubov Transformation Functions
double uk(double eps_k, double E_k);                          //                            real-valued Bogoliubov constant
double vk(double eps_k, double f_E_k, double Deltak);                           //                            real-valued Bogoliubov constant (because there is no initial phase)
double Rk(double epsk, double u_k, double v_k, dcomp Delta_c);       // meV                    real-valued SC Bogoliubov Hamiltonian parameter
dcomp Ck(double epsk, double u_k, double v_k, dcomp Delta_c);       // meV                     complex-valued SC Bogoliubov Hamiltonian parameter

// Simplifying functions for the Hamiltonian
double Lp(double uk, double ukq, double vk, double vkq);                                                    //                              real-valued (because there is no intitial phase)
double Lm(double uk, double ukq, double vk, double vkq);                                                   //                              real-valued (because there is no intitial phase)
double Mp(double uk, double ukq, double vk, double vkq);                                                   //                              real-valued (because there is no intitial phase)
double Mm(double uk, double ukq, double vk, double vkq);                                                  //                              real-valued (because there is no intitial phase)

// Pump Functions
dcomp Aqp(double t);                                                     //                              for positive q: complex-valued EM Hamiltonian pump parameter
dcomp Aqm(double t);                                                    //                              for negative q
dcomp Aq_pr(double t);

// Help Functions
double kiTok(int ki, int itheta1);                                                        //  a                          wave vector from wave number
double kCurrent(int ki, int itheta);
double deltaf(int k1, int k2);                                            //                            calculate \delta : return 1.0 if k1 = k2;

// Main Variables & Time
dcomp Delta = 0.0;
double t = 0.0; 
double dtmax;
double dt_p;
double dt_pr;
double tstart;




// Define Momentum Arrays
// arrays for the current value of the quasiparticle expectation value: y_n
dcomp * aDa;                                                          //                          alpha-dagger alpha array
dcomp * bDb;                                                        //                          beta-dagger beta array
dcomp * aDbD;                                                       //                          alpha-dagger beta-dagger array
dcomp * ab;                                                            //                          alpha beta array

// arrays for storing the aggregate value of the current Runge-Kutta step, i: y_{n+1} = y_n + sum_i{k_i}
dcomp * aDa_rk;
dcomp * bDb_rk;
dcomp * aDbD_rk;
dcomp * ab_rk;

// arrays for storing the current value of the quasiparticles for the next Runge-Kutta step, i: y_n + C*k_i where C is some constant, 1.0 or 0.5;
dcomp * aDa_k;
dcomp * bDb_k;
dcomp * aDbD_k;
dcomp * ab_k;

dcomp * aDa_ddt;
dcomp * bDb_ddt;
dcomp * aDbD_ddt;
dcomp * ab_ddt;

dcomp * aDbD_pr;
dcomp * ab_pr;
dcomp * aDa_pr;
dcomp * bDb_pr;

dcomp * aDbD_ddt_pr;
dcomp * ab_ddt_pr;
dcomp * aDa_ddt_pr;
dcomp * bDb_ddt_pr;

dcomp * aDbD_k_pr;
dcomp * ab_k_pr;
dcomp *aDa_k_pr;
dcomp *bDb_k_pr;

dcomp * aDbD_rk_pr;
dcomp * ab_rk_pr;
dcomp * aDa_rk_pr;
dcomp * bDb_rk_pr;

// Functions for the Differential Equation
// precalculates all the variables for a given k, k' and t. ie. for a single entry in each of the four matrices for a given time.
// then calculates the entry four each of the four matrices for the given time, using the following differential equations.
// finally, calcualtes and returns the value of the gap at the current time iteration
void diffEq(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], dcomp ab_c[]);
void diffEq2(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[]);
void diffEq3(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], 
 dcomp aDa_pr[], dcomp bDb_pr[], dcomp aDbD_pr[], dcomp ab_pr[]);
void diffEq4(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], 
 dcomp aDa_pr[], dcomp bDb_pr[], dcomp aDbD_pr[], dcomp ab_pr[]);

// Current response
dcomp currentEq(double t, int idt, dcomp aDa_c_pr[], dcomp bDb_c_pr[], dcomp ab_c_pr[], dcomp aDbD_c_pr[]);

// Runge-Kutta 4th Order implementation for the Bogoliubov quasiparticle expectation values
void RungeKutta(double t, double h);
void RungeKutta2(double t, double h);
void RungeKutta3(double t, double h);
void RungeKutta4(double t, double h);

void reformDiagArrays();

//Calculate W for zero temperature
void calcW();

// Function to Calculate the Gap Function (For first and last step)
dcomp gapEq(dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], dcomp ab_c[]);
dcomp gapEq2(dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[]);

// Function to calculate the angel dependent global variables
// Function to calculate the angel dependent global variables
void calcAngles();
double intToAngle(int itheta);
void arrayLengths();
double thetac;

double get_wall_time();

// Dwave Help Functions
dcomp calcDeltak(dcomp Delta_c, int itheta1, int ki);
double calcDelta0k(int itheta1, int ki);
double calcAnglek(int itheta1, int ki);
double dwave(double theta);

double calcdt(int idt);

// Sort Help Functions
bool kSort(int i, int j);
void arrayIorder(int a[],  int idx[]);
void arrayDorder(double a[],  int idx[]);

// Function for indexing a 3D Matrix in my 1D c++ array
int index_p(int itheta, int ik, int idiag);
int index_diag(int itheta, int ik);
int index_pr(int itheta, int idt, int ik);


int main()
{
    int i, j, itheta1, Ntotal, idt;
    double theta1, theta2, check_epsk, check_Ek, check_vk, check_uk, check_i, check_Mm, check_Mp, check_Lm, check_Lp, check_R, iterationtime;
    clock_t start;
    dcomp check_C;

    // Initialize Delta symmetry on the FS
    //    for (itheta1 = 0; i<Ntheta1; itheta1++){
    //        theta1 = intToAngle(itheta1);

    //        Deltak0[itheta1] = Delta0*dwave(theta1);
    //    }
    dt_p = round(tau_p/2.0*sqrt(log(1000.0)/log(2.0)));
	tstart = -dt_p;
    dt_pr = round(tau_pr/2.0*sqrt(log(1000.0)/log(2.0)));
    dtmax = calcdt(Ndt-1);

    printf("Calculating Matrices \n");
    // Initialize all the FS variables
    calcAngles();

    // Calculate total length of arrays
    Ntotal = 0;
    N_pr = 0;
    N_diag = 0;
    for (itheta1 = 0; itheta1<Ntheta1; itheta1++){
        Ntotal = Ntotal+N[itheta1]*Ndiag;
        N_diag = N_diag+N[itheta1];
		N_pr = N_pr+N[itheta1]*Ndt;
    }


    // Initialize Pump Momentum Arrays
    // arrays for the current value of the quasiparticle expectation value: y_n
    aDa = new dcomp[N_diag]();                                                           //                          alpha-dagger alpha array
    bDb = new dcomp[N_diag]();                                                           //                          beta-dagger beta array
    aDbD = new dcomp[N_diag]();                                                       //                          alpha-dagger beta-dagger array
//     ab = new dcomp[Ntotal]();                                                             //                          alpha beta array

    // arrays for storing the aggregate value of the current Runge-Kutta step, i: y_{n+1} = y_n + sum_i{k_i}
    aDa_rk = new dcomp[N_diag];                                                           //                          alpha-dagger alpha array
    bDb_rk = new dcomp[N_diag];                                                           //                          beta-dagger beta array
    aDbD_rk = new dcomp[N_diag];                                                       //                          alpha-dagger beta-dagger array
//     ab_rk = new dcomp[Ntotal];                                                             //                          alpha beta array

    // arrays for storing the current value of the quasiparticles for the next Runge-Kutta step, i: y_n + C*k_i where C is some constant, 1.0 or 0.5;
    aDa_k = new dcomp[N_diag];                                                           //                          alpha-dagger alpha array
    bDb_k = new dcomp[N_diag];                                                           //                          beta-dagger beta array
    aDbD_k = new dcomp[N_diag];                                                       //                          alpha-dagger beta-dagger array
//     ab_k = new dcomp[Ntotal];                                                             //                          alpha beta array

    aDa_ddt = new dcomp[N_diag];
    bDb_ddt = new dcomp[N_diag];
    aDbD_ddt = new dcomp[N_diag];
//     ab_ddt = new dcomp[Ntotal];
    
    // Initialize Probe Momentum Arrays
    // arrays for the current value of the quasiparticle expectation value: y_n
    aDa_pr = new dcomp[N_pr]();                                                           //                          alpha-dagger alpha array
    bDb_pr = new dcomp[N_pr]();                                                           //                          beta-dagger beta array
    aDbD_pr = new dcomp[N_pr]();                                                       //                          alpha-dagger beta-dagger array
    ab_pr = new dcomp[N_pr]();                                                             //                          alpha beta array

    // arrays for storing the aggregate value of the current Runge-Kutta step, i: y_{n+1} = y_n + sum_i{k_i}
    aDa_rk_pr = new dcomp[N_pr]();                                                           //                          alpha-dagger alpha array
    bDb_rk_pr = new dcomp[N_pr]();                                                           //                          beta-dagger beta array
    aDbD_rk_pr = new dcomp[N_pr]();                                                       //                          alpha-dagger beta-dagger array
    ab_rk_pr = new dcomp[N_pr]();                                                             //                          alpha beta array

    // arrays for storing the current value of the quasiparticles for the next Runge-Kutta step, i: y_n + C*k_i where C is some constant, 1.0 or 0.5;
    aDa_k_pr = new dcomp[N_pr]();                                                           //                          alpha-dagger alpha array
    bDb_k_pr = new dcomp[N_pr]();                                                           //                          beta-dagger beta array
    aDbD_k_pr = new dcomp[N_pr]();                                                       //                          alpha-dagger beta-dagger array
    ab_k_pr = new dcomp[N_pr]();                                                             //                          alpha beta array

    aDa_ddt_pr = new dcomp[N_pr]();
    bDb_ddt_pr = new dcomp[N_pr]();
    aDbD_ddt_pr = new dcomp[N_pr]();
    ab_ddt_pr = new dcomp[N_pr]();
    
    printf("Matrices produced \n");
    
    printf("Reading in Data \n");
    // Input the Data from the Matrices.txt file
    double in1,in2,in3,in4,in5,in6, in7, in8;
    ifstream matricesIn("data_Matrices.txt");
    if (matricesIn.is_open()){
        i = 0;
        while(matricesIn >> in1 >> in2 >> in3 >> in4 >> in5 >> in6){
            aDa[i] = dcomp(in1,in2);
            bDb[i] = dcomp(in3,in4);
            aDbD[i] = dcomp(in5,in6);
//             ab[i] = dcomp(in7,in8);
            ++i;
        }
    }
//     matricesIn.close();
//     
//     unsigned int num_of_lines = 0;
//     string line;
//     ifstream deltaIn("Delta.txt");
//     if(deltaIn.is_Open()){
//         while(getline(deltaIn,line)
//         ++num_of_lines;
//     }
//     dataDelta = new dcomp[num_of_lines];
//     i = 0;
//     while(deltaIn >> in1 >> in2 >> in3){
//         dataDelta[i] = dcomp(in2,in3);
//         ++i;
//         
//     }
    
    ifstream endIn("data_End.txt");
    if (endIn.is_open()){
        while(endIn >> t >> in1 >> in2){
            Delta = dcomp(in1,in2);
        }
    }
    endIn.close();
    printf("Data read \n");
    
//     ifstream deltaIn("data_Delta.txt");
//     if (deltaIn.is_open())
//         while(deltaIn
    
        
    FILE *timing;
    timing = fopen("IterationTime.txt","w");
    FILE *conditionsOutput;
    conditionsOutput = fopen("Parameters.txt","w");
    FILE *aDaOut;
    aDaOut = fopen("data_Matrices.txt","w");
    FILE *deltaOut;
    deltaOut = fopen("Delta.txt","w");
    FILE *data_Delta;
    data_Delta = fopen("data_Delta.txt","w");
    FILE *currentR;
    currentR = fopen("currentR.txt","w");
	FILE * currentI;
    currentI = fopen("currentI.txt","w");
    FILE * data_End; 
    data_End = fopen("data_End.txt","w");
    
    
    theta2 = 0.0;

    Delta = gapEq2(aDa, bDb, aDbD);
    fprintf(deltaOut, "%e\n", abs(Delta));
    fprintf(data_Delta, "%e\t", real(Delta));
    fprintf(data_Delta, "%e\n", imag(Delta));
    fclose(data_Delta);
	
	fprintf(conditionsOutput, "Properties of the Superconductor\n");
	fprintf(conditionsOutput, "d-wave symmetry \n");
	fprintf(conditionsOutput, "Initial gap: %f\n", abs(Delta));
	fprintf(conditionsOutput, "Fermi energy Ef: %f\n", Ef);
	fprintf(conditionsOutput, "Mass m: %f\n", m);
	fprintf(conditionsOutput, "Lattice constant a: %e\n", a);
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
	fprintf(conditionsOutput, "Run Time: %f\n", tend);
	fprintf(conditionsOutput, "Time Step Size h: %f\n", h);
	fprintf(conditionsOutput, "Probe delay time: %f\n", dtmin);
	fprintf(conditionsOutput, "Probe time step hdt: :q%f\n", hdt);
    
    fclose(conditionsOutput);

	double dt;
	dcomp currentDensity;
    
    if(tstart>tend)
        return 0;
	// DURING THE PUMP
// 	printf("During the pump \n");
//     for (t = t; t<dt_p; t+= h){
//         if(t>tend)
//             break;
//         start = get_wall_time();
// 
//         RungeKutta(t,h);
//         fprintf(deltaOut, "%e\n", abs(Delta));
//         fprintf(data_Delta, "%e\t", real(Delta));
//         fprintf(data_Delta, "%e\n", imag(Delta));
// 
//         iterationtime = (get_wall_time()-start);
//         fprintf(timing, "%f\n", iterationtime);
// 	}
 
//     reformDiagArrays(); //Comment for tests
//     i = int(t-tstart)-1;
    if(t<tend){
		// BETWEEN THE PUMP AND PROBE
		printf("Between the pump and probe \n");
		for (t = t; t<tend; t+= h){
            start = get_wall_time();
            
//             Delta = dataDelta[i];
            RungeKutta2(t,h);
            fprintf(deltaOut, "%e\n", abs(Delta));

			iterationtime = (get_wall_time()-start);
			fprintf(timing, "%f\n", iterationtime);
		}
	}
//     if(t<tend){
// 		// DURING THE PROBE
// 		printf("During the probe \n");
// 		for (t = t; t<dtmax+dt_pr; t+= h){
// 			start = get_wall_time();
// 		
//             RungeKutta3(t,h);
// //             fprintf(deltaOut, "%e\n  ", abs(Delta));
// //             fprintf(data_Delta, "%e\t", real(Delta));
// //             fprintf(data_Delta, "%e\n", imag(Delta));
// 					
// 			for (idt = 0; idt<Ndt; idt++){
// 				dt = calcdt(idt);
// 				if (t>dt-dt_pr){
// 					currentDensity = currentEq(t, idt, aDa_pr, bDb_pr, ab_pr, aDbD_pr);
// 					fprintf(currentR, "%e\t", real(currentDensity));
// 					fprintf(currentI, "%e\t", imag(currentDensity));
// 				}
// 			}
// 		
// 			fprintf(currentR, "\n");
// 			fprintf(currentI, "\n");
// 			
// 			iterationtime = (get_wall_time()-start);
// 			fprintf(timing, "%f\n", iterationtime);
// 		}
// 	}	
//     if(t<tend){
// 		// AFTER THE PROBE
// 		printf("After the probe \n");
// 		for (t = t; t<tend+dtmax; t+= h){
// 			start = get_wall_time();
// 		
//             RungeKutta4(t,h);
// //             fprintf(deltaOut, "%e\n", abs(Delta));
// //             fprintf(data_Delta, "%e\t", real(Delta));
// //             fprintf(data_Delta, "%e\n", imag(Delta));
// 			
// 			
// 			for (idt = 0; idt<Ndt-1; idt++){
// 				dt = calcdt(idt);
// 				if(t<tend+dt){
// 					currentDensity = currentEq(t, idt, aDa_pr, bDb_pr, ab_pr, aDbD_pr);
// 					fprintf(currentR, "%e\t", real(currentDensity));
// 					fprintf(currentI, "%e\t", imag(currentDensity));
// 				}
// 				else{
// 					fprintf(currentR, "\t");
// 					fprintf(currentI, "\t");
// 				}
// 			}
// 			
// 			currentDensity = currentEq(t, Ndt-1, aDa_pr, bDb_pr, ab_pr, aDbD_pr);
// 			fprintf(currentR, "%e\n", real(currentDensity));
// 			fprintf(currentI, "%e\n", imag(currentDensity));
// 			
// 			iterationtime = (get_wall_time()-start);
// 			fprintf(timing, "%f\n", iterationtime);
// 		}
// 	}
    
    printf("Writing Matrices \n");
    
    // Close Output Files
    fclose(currentR);
    fclose(currentI);
    fclose(deltaOut);
    fclose(timing);
    
    for (i = 0; i<Ndiag; i++){ // print final values
        fprintf(aDaOut, "%e\t", real(aDa[i]));
        fprintf(aDaOut, "%e\t", imag(aDa[i]));
        fprintf(aDaOut, "%e\t", real(bDb[i]));
        fprintf(aDaOut, "%e\t", imag(bDb[i]));
        fprintf(aDaOut, "%e\t", real(aDbD[i]));
        fprintf(aDaOut, "%e\n", imag(aDbD[i]));
    }
    fclose(aDaOut);
    
    fprintf(data_End, "%f\t", t);
    fprintf(data_End, "%e\t", real(Delta));
    fprintf(data_End, "%e\n", imag(Delta));
    fclose(data_End);
    // Delete dynmaic variables
    
    delete[] aDa;
    delete[] bDb;
    delete[] aDbD;

    delete[] aDa_k;
    delete[] bDb_k;
    delete[] aDbD_k;

    delete[] aDa_rk;
    delete[] bDb_rk;
    delete[] aDbD_rk;

    delete[] aDa_ddt;
    delete[] bDb_ddt;
    delete[] aDbD_ddt;
    
	delete[] aDa_pr;
    delete[] bDb_pr;
    delete[] aDbD_pr;

    delete[] aDa_k_pr;
    delete[] bDb_k_pr;
    delete[] aDbD_k_pr;
    delete[] ab_k_pr;

    delete[] aDa_rk_pr;
    delete[] bDb_rk_pr;
    delete[] aDbD_rk_pr;
    delete[] ab_rk_pr;

    delete[] aDa_ddt_pr;
    delete[] bDb_ddt_pr;
    delete[] aDbD_ddt_pr;
    delete[] ab_ddt_pr;

    return 0;
}

/////////////////////////////////////////////////////////////////
//             Help Functions
//////////////////////////////////////////////////////////////////

// Calculate the wave vector squared from the integer wave number 
// k^2 !!!!!!! I NEED k...........
double kiTok(int ki, int itheta1){
    return (pow((double(ki)*q0+kx0[itheta1]),2.0)+pow(kFy[itheta1],2.0));
}

// Calculate the wave vector in the x direction form the integer wave number (also in the x direction)
// this is for k not k^2...!
double kCurrent(int ki, int itheta){
	return double(ki)*q0+kx0[itheta];
}


// calculate \delta : return 1.0 if k1 = k2;
double deltaf(int k1, int k2){
    double output = 0.0;

    if (k1 == k2){
        output = 1.0;
    }

    return output;
}

////////////////////////////////////////////////////////////////
//              Energy Dispersions
////////////////////////////////////////////////////////////////

// Band dispersion energies

// band dispersion energy
// input the wave number ki in the x-direction only
double epsK(int ki, int itheta1){
    double k;

    k = kiTok(ki, itheta1);
    return t0*k-Ef;
}

double epsK_pr(int ki, int itheta1){
    double k = (pow((double(ki)*q0+q_pr+kx0[itheta1]),2.0)+pow(kFy[itheta1],2.0));

    return t0*k-Ef;
}

// SC quasiparticle energy
double Ek(double eps_k, double Deltak){
    double test; // test if zero
    test = sqrt(pow(eps_k,2.0)+pow(Deltak,2.0)); // Deltak must be k-dependent
    return test;
}
///////////////////////////////////////////////////////////////////
//       Bogoliubov Transformation variables
///////////////////////////////////////////////////////////////////

// real-valued Bogoliubov variable u_k
double uk(double eps_k, double E_k){
    return sqrt(0.5*(1.0+eps_k/E_k));
}

// real-valued Bogoliubov variable v_k (because there is no initial phase)
double vk(double eps_k, double E_k, double Deltak){
    return copysign(sqrt(0.5*(1.0-eps_k/E_k)), Deltak);
}

// real-valued Bogoliubov variable R_k
double Rk(double epsk, double u_k, double v_k, dcomp Delta){
    return epsk*(1.0-2.0*pow(v_k,2.0))+2.0*real(Delta)*u_k*v_k;
}

// complex-valued Bogoliubov variable C_k
dcomp Ck(double epsk, double u_k, double v_k, dcomp Delta){
    return -2.0*epsk*u_k*v_k+Delta*pow(u_k,2.0)-conj(Delta)*pow(v_k,2.0);
}

// L_{k,q'}^{+}
double Lp(double uk, double ukq, double vk, double vkq){
    return uk*ukq+vk*vkq;
}

// L_{k,q'}^{-}
double Lm(double uk, double ukq, double vk, double vkq){
    return uk*ukq-vk*vkq;
}

// M_{k,q'}^{+}
double Mp(double uk, double ukq, double vk, double vkq){
    return vk*ukq+uk*vkq;
}

// M_{k,q'}^{-}
double Mm(double uk, double ukq, double vk, double vkq){
    return vk*ukq-uk*vkq;
}

////////////////////////////////////////////////////////////////
//                  Pump Pulse
////////////////////////////////////////////////////////////////
// calculated for the plus/minus delta term (p,m) without A0 (included in A0eff1 and A0eff2);

// q  = +q0
dcomp Aqp(double t){

    return exp(-pow(2.0*sqrt(log(2.0))*t/tau_p,2.0))*exp(-I*w0_p*t);

}

// q = -q0
dcomp Aqm(double t){

    return exp(-pow(2.0*sqrt(log(2.0))*t/tau_p,2.0))*exp(I*w0_p*t);

}

dcomp Aq_pr(double tminusdt){ // NEW: checking in the differential equation
//    if (t>(dt-dt_pr) %% t<(dt+dt_pr)){
    return exp(-pow(2.0*sqrt(log(2.0))*(tminusdt)/tau_pr,2.0))*exp(-I*w0_pr*(tminusdt));
//    }
//    else
//        return 0.0;
}

//////////////////////////////////////////////////////////////
//                Equations of Motion
//////////////////////////////////////////////////////////////
// Calculates the equations of motions for all the entries in the arrays
// Also calculates the term in the sum of the gap function for the given input value of ki
// returns the Gap value calculated for the sum over all ki points in the array

void diffEq(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], dcomp ab_c[]){
    // input time, matrices indices, matrices for calculations

    // Local variables for the momentum k = ki,  k' = ki + ji entry (ji is one of {-4,-3,-2,-1,0,1,2,3,4}
    // _m2 indicates q' = -2*q0, _m1 is q' = -q0, _0 is q0 = 0; _p1 is q' = q0; _p2 is q' = 2*q0
    // variabels to be changed at each time step (pre-calculated before the calculating the result of the differential equation)
    double epsk_m2, epsk_m1, epsk_0, epsk_p1, epsk_p2, epskp_m2, epskp_m1, epskp_0, epskp_p1, epskp_p2,
            Ek_m2, Ek_m1, Ek_0, Ek_p1, Ek_p2, Ekp_m2, Ekp_m1, Ekp_0, Ekp_p1, Ekp_p2,
            uk_m2, uk_m1, uk_0, uk_p1, uk_p2, ukp_m2, ukp_m1, ukp_0, ukp_p1, ukp_p2,
            vk_m2, vk_m1, vk_0, vk_p1, vk_p2, vkp_m2, vkp_m1, vkp_0, vkp_p1, vkp_p2,
            R_k, R_kp, Deltak_m2, Deltak_m1, Deltak_0, Deltak_p1, Deltak_p2,
            Deltakp_m2, Deltakp_m1, Deltakp_0, Deltakp_p1, Deltakp_p2,
            Lpk_m1, Lpk_p1, Lmk_m2, Lmk_0, Lmk_p2,
            Lpkp_m1, Lpkp_p1, Lmkp_m2, Lmkp_0, Lmkp_p2,
            Mpk_m2, Mpk_0, Mpk_p2, Mmk_m1, Mmk_p1,
            Mpkp_m2, Mpkp_0, Mpkp_p2, Mmkp_m1, Mmkp_p1;
    dcomp Delta_ck, Delta_ckp, C_k, C_kp, Aq_p1, Aq_m1, t1, t2, t3, t4, t5, t6;
    int ji, ki, kpi, kpdiag, kdiag, krow, kprow, i, ip;

    // Initialize ddt arrays
    //    aDa_ddt = new dcomp[N[itheta1]*Ndiag]();
    //    bDb_ddt = new dcomp[N[itheta1]*Ndiag]();
    //    aDbD_ddt = new dcomp[N[itheta1]*Ndiag]();
    //    ab_ddt = new dcomp[N[itheta1]*Ndiag]();

    //    int kpi = ki+ji;
    // Calculate Aq_p1, Aq_m1

    // during the pulse

    Aq_p1 = Aqp(t);
    Aq_m1 = Aqm(t);

    for (ki = 0; ki<Nk[itheta1]; ki++){
        // Pre-calculate all the variables needed for the differential equation for variable k:
        krow = ki+arrayBuffer;

        // calculate epsk for q' = -2q0, ..., 2q0,
        epsk_m2 = epsK(ki-2, itheta1); epsk_m1 = epsK(ki-1, itheta1); epsk_0= epsK(ki, itheta1); epsk_p1 = epsK(ki+1, itheta1), epsk_p2 = epsK(ki+2, itheta1);

        // calculate Delta0(k) for q' = -2q0, ..., 2q0,
        Deltak_m2 = calcDelta0k(itheta1, ki-2); Deltak_m1 = calcDelta0k(itheta1, ki-1); Deltak_0 = calcDelta0k(itheta1,ki);
        Deltak_p1 = calcDelta0k(itheta1, ki+1); Deltak_p2 = calcDelta0k(itheta1, ki+2);

        // calculate Ek for q' = -2q0, ..., 2q0,
        Ek_m2 = Ek(epsk_m2, Deltak_m2); Ek_m1 = Ek(epsk_m1, Deltak_m1); Ek_0 = Ek(epsk_0, Deltak_0); Ek_p1 = Ek(epsk_p1, Deltak_p1); Ek_p2 = Ek(epsk_p2, Deltak_p2);

        // calculate uk for q' = -2q0, ..., 2q0,
        uk_m2 = uk(epsk_m2, Ek_m2); uk_m1 = uk(epsk_m1, Ek_m1); uk_0 = uk(epsk_0, Ek_0); uk_p1 = uk(epsk_p1, Ek_p1); uk_p2 = uk(epsk_p2, Ek_p2);

        // calculate vk for q' = -2q0, ..., 2q0,
        vk_m2 = vk(epsk_m2, Ek_m2, Deltak_m2); vk_m1 = vk(epsk_m1, Ek_m1, Deltak_m1); vk_0 = vk(epsk_0, Ek_0, Deltak_0); vk_p1 = vk(epsk_p1, Ek_p1, Deltak_p1); vk_p2 = vk(epsk_p2, Ek_p2, Deltak_p2);

        // Calculate Delta_c(k)
        Delta_ck = calcDeltak(Delta_c, itheta1, ki);

        // calculate Ck, Rk, Ckp, Rkp
        C_k = Ck(epsk_0,uk_0,vk_0,Delta_ck); R_k = Rk(epsk_0,uk_0,vk_0,Delta_ck);

        // Calculate L_{k,q'}^{+} for q' = -2q0, ..., 2q0
        Lpk_m1 = uk_0*uk_m1+vk_0*vk_m1;
        Lpk_p1 = uk_0*uk_p1+vk_0*vk_p1;

        // Calculate L_{k,q'}^{-} for q' = -2q0, ..., 2q0
        Lmk_m2 = uk_0*uk_m2-vk_0*vk_m2; Lmk_0 = uk_0*uk_0-vk_0*vk_0;
        Lmk_p2 = uk_0*uk_p2-vk_0*vk_p2;

        // Calculate M_{k,q'}^{+} for q' = -2q0, ..., 2q0
        Mpk_m2 = vk_0*uk_m2+uk_0*vk_m2; Mpk_0 = vk_0*uk_0+uk_0*vk_0;
        Mpk_p2 = vk_0*uk_p2+uk_0*vk_p2;

        // Calculate M_{k,q'}^{-} for q' = -2q0, ..., 2q0
        Mmk_m1 = vk_0*uk_m1-uk_0*vk_m1;
        Mmk_p1 = vk_0*uk_p1-uk_0*vk_p1;

        for (ji = -4; ji<5; ji++){
            if (ji + ki>=0 && ji + ki < Nk[itheta1]){
                kpi = ki+ji;
                kprow = krow+ji;
                kdiag = ji+6;
                kpdiag = -ji+6;
                i = index_p(itheta1, krow, kdiag);
                ip = index_p(itheta1, kprow, kpdiag);
                // Pre-calculate all the kp functions

                // Calculate epskp for q' = -2q0, ..., 2q0
                epskp_m2 = epsK(kpi-2, itheta1); epskp_m1 = epsK(kpi-1, itheta1); epskp_0 = epsK(kpi, itheta1); epskp_p1 = epsK(kpi+1, itheta1), epskp_p2 = epsK(kpi+2, itheta1);

                // calculate Delta0(kp) for q' = -2q0, ..., 2q0,
                Deltakp_m2 = calcDelta0k(itheta1, kpi-2); Deltakp_m1 = calcDelta0k(itheta1, kpi-1); Deltakp_0 = calcDelta0k(itheta1,kpi);
                Deltakp_p1 = calcDelta0k(itheta1, kpi+1); Deltakp_p2 = calcDelta0k(itheta1, kpi+2);

                // Calculate Ekp for q' = -2q0, ..., 2q0
                Ekp_m2 = Ek(epskp_m2, Deltakp_m2); Ekp_m1 = Ek(epskp_m1, Deltakp_m1); Ekp_0 = Ek(epskp_0, Deltakp_0); Ekp_p1 = Ek(epskp_p1, Deltakp_p1); Ekp_p2 = Ek(epskp_p2, Deltakp_p2);

                // Calculate ukp for q' = -2q0, ..., 2q0
                ukp_m2 = uk(epskp_m2, Ekp_m2); ukp_m1 = uk(epskp_m1, Ekp_m1); ukp_0 = uk(epskp_0, Ekp_0); ukp_p1 = uk(epskp_p1, Ekp_p1); ukp_p2 = uk(epskp_p2, Ekp_p2);

                // Calculate vkp for q' = -2q0, ..., 2q0
                vkp_m2 = vk(epskp_m2, Ekp_m2, Deltakp_m2); vkp_m1 = vk(epskp_m1, Ekp_m1, Deltakp_m1); vkp_0 = vk(epskp_0, Ekp_0, Deltakp_0); vkp_p1 = vk(epskp_p1, Ekp_p1, Deltakp_p1); vkp_p2 = vk(epskp_p2, Ekp_p2, Deltakp_p2);

                // Calculate Delta_c(kp)
                Delta_ckp = calcDeltak(Delta_c, itheta1, kpi);

                // Calculate Ckp, Rkp
                C_kp = Ck(epskp_0,ukp_0,vkp_0,Delta_ckp); R_kp = Rk(epskp_0,ukp_0,vkp_0,Delta_ckp);

                // Calculate L_{k',q'}^{+} for q' = -2q0, ..., 2q0
                Lpkp_m1 = ukp_0*ukp_m1+vkp_0*vkp_m1;
                Lpkp_p1 = ukp_0*ukp_p1+vkp_0*vkp_p1;

                // Calculate L_{k',q'}^{-} for q' = -2q0, ..., 2q0
                Lmkp_m2 = ukp_0*ukp_m2-vkp_0*vkp_m2; Lmkp_0 = ukp_0*ukp_0-vkp_0*vkp_0;
                Lmkp_p2 = ukp_0*ukp_p2-vkp_0*vkp_p2;

                // Calculate M_{k',q'}^{+} for q' = -2q0, ..., 2q0
                Mpkp_m2 = vkp_0*ukp_m2+ukp_0*vkp_m2; Mpkp_0 = vkp_0*ukp_0+ukp_0*vkp_0;
                Mpkp_p2 = vkp_0*ukp_p2+ukp_0*vkp_p2;

                // Calculate M_{k',q'}^{-} for q' = -2q0, ..., 2q0
                Mmkp_m1 = vkp_0*ukp_m1-ukp_0*vkp_m1;
                Mmkp_p1 = vkp_0*ukp_p1-ukp_0*vkp_p1;

                //
                // Carry out the differential equation RHS for equation 1-4
                //

                // first equation:

                //first term
                t1 = -(R_k+R_kp)*aDbD_c[i]+conj(C_kp)*aDa_c[i]+conj(C_k)*(bDb_c[ip]-deltaf(kpi,ki));

                // first sum qp = +/- q0
                // qp = + q0
                t2 = Aq_p1*(-Lpk_p1*aDbD_c[index_p(itheta1, krow+1, kdiag-1)]+Lpkp_m1*aDbD_c[index_p(itheta1, krow, kdiag-1)]
                        -Mmkp_m1*aDa_c[index_p(itheta1, krow, kdiag-1)]+Mmk_p1*(bDb_c[index_p(itheta1, kprow, kpdiag+1)]-deltaf(kpi-ki,1)));

                // qp = - q0
                t3 = Aq_m1*(-Lpk_m1*aDbD_c[index_p(itheta1, krow-1, kdiag+1)]+Lpkp_p1*aDbD_c[index_p(itheta1, krow, kdiag+1)]
                        -Mmkp_p1*aDa_c[index_p(itheta1, krow, kdiag+1)]+Mmk_m1*(bDb_c[index_p(itheta1, kprow, kpdiag-1)]-deltaf(kpi-ki,-1)));

                // second sum qp = +/- 2q0, 0
                // qp = -2 q0
                t4 = pow(Aq_m1,2.0)*(-Lmk_m2*aDbD_c[index_p(itheta1, krow-2, kdiag+2)]-Lmkp_p2*aDbD_c[index_p(itheta1, krow, kdiag+2)]
                        -Mpkp_p2*aDa_c[index_p(itheta1, krow, kdiag+2)]+Mpk_m2*(-bDb_c[index_p(itheta1, kprow, kpdiag-2)]+deltaf(kpi-ki,-2)));

                // qp = -0
                t5 = 2.0*Aq_p1*Aq_m1*(-Lmk_0*aDbD_c[i]-Lmkp_0*aDbD_c[i]
                        -Mpkp_0*aDa_c[i]+Mpk_0*(-bDb_c[ip]+deltaf(kpi,ki)));

                // qp = +2 q0
                t6 = pow(Aq_p1,2.0)*(-Lmk_p2*aDbD_c[index_p(itheta1, krow+2, kdiag-2)]-Lmkp_m2*aDbD_c[index_p(itheta1, krow, kdiag-2)]
                        -Mpkp_m2*aDa_c[index_p(itheta1, krow, kdiag-2)]+Mpk_p2*(-bDb_c[index_p(itheta1, kprow, kpdiag+2)]+deltaf(kpi-ki,+2)));

                aDbD_ddt[i] = -I/hbar*(t1+kdotA[itheta1]*(t2+t3)+A0eff2*(t4+t5+t6));

                // second equation:

                //first term
                t1 = (R_k+R_kp)*ab_c[i]+C_kp*aDa_c[ip]+C_k*(bDb_c[i]-deltaf(kpi,ki));

                // first sum qp = +/- q0
                // qp = + q0
                t2 = Aq_m1*(Lpk_p1*ab_c[index_p(itheta1, krow+1, kdiag-1)]-Lpkp_m1*ab_c[index_p(itheta1, krow, kdiag-1)]
                        -Mmkp_m1*aDa_c[index_p(itheta1, kprow-1, kpdiag+1)]+Mmk_p1*(bDb_c[index_p(itheta1, krow+1, kdiag-1)]-deltaf(kpi-ki,1)));

                // qp = - q0
                t3 = Aq_p1*(Lpk_m1*ab_c[index_p(itheta1, krow-1, kdiag+1)]-Lpkp_p1*ab_c[index_p(itheta1, krow, kdiag+1)]
                        -Mmkp_p1*aDa_c[index_p(itheta1, kprow+1, kpdiag-1)]+Mmk_m1*(bDb_c[index_p(itheta1, krow-1, kdiag+1)]-deltaf(kpi-ki,-1)));

                // second sum qp = +/- 2q0, 0
                // qp = -2 q0
                t4 = pow(Aq_p1,2.0)*(Lmk_m2*ab_c[index_p(itheta1, krow-2, kdiag+2)]+Lmkp_p2*ab_c[index_p(itheta1, krow, kdiag+2)]
                        -Mpkp_p2*aDa_c[index_p(itheta1, kprow+2, kpdiag-2)]+Mpk_m2*(-bDb_c[index_p(itheta1, krow-2, kdiag+2)]+deltaf(kpi-ki,-2)));

                // qp = -0
                t5 = 2.0*Aq_p1*Aq_m1*(Lmk_0*ab_c[i]+Lmkp_0*ab_c[i]
                        -Mpkp_0*aDa_c[ip]+Mpk_0*(-bDb_c[i]+deltaf(kpi,ki)));

                // qp = +2 q0
                t6 = pow(Aq_m1,2.0)*(Lmk_p2*ab_c[index_p(itheta1, krow+2, kdiag-2)]+Lmkp_m2*ab_c[index_p(itheta1, krow, kdiag-2)]
                        -Mpkp_m2*aDa_c[index_p(itheta1, kprow-2, kpdiag+2)]+Mpk_p2*(-bDb_c[index_p(itheta1, krow+2, kdiag-2)]+deltaf(kpi-ki,2)));


                ab_ddt[i] = -I/hbar*(t1+kdotA[itheta1]*(t2+t3)+A0eff2*(t4+t5+t6));

                // third equation:

                //first term
                t1 = (R_kp-R_k)*aDa_c[i]+C_kp*aDbD_c[i]+conj(C_k)*ab_c[ip];

                // first sum qp = +/- q0
                // qp = + q0
                t2 = Aq_p1*(-Lpk_p1*aDa_c[index_p(itheta1, krow+1, kdiag-1)]+Lpkp_m1*aDa_c[index_p(itheta1, krow, kdiag-1)]
                        +Mmk_p1*ab_c[index_p(itheta1, kprow, kpdiag+1)]+Mmkp_m1*aDbD_c[index_p(itheta1, krow, kdiag-1)]);

                // qp = - q0
                t3 = Aq_m1*(-Lpk_m1*aDa_c[index_p(itheta1, krow-1, kdiag+1)]+Lpkp_p1*aDa_c[index_p(itheta1, krow, kdiag+1)]
                        +Mmk_m1*ab_c[index_p(itheta1, kprow, kpdiag-1)]+Mmkp_p1*aDbD_c[index_p(itheta1, krow, kdiag+1)]);

                // second sum qp = +/- 2q0, 0
                // qp = -2 q0
                t4 = pow(Aq_m1,2.0)*(-Lmk_m2*aDa_c[index_p(itheta1, krow-2, kdiag+2)]+Lmkp_p2*aDa_c[index_p(itheta1, krow, kdiag+2)]
                        -Mpk_m2*ab_c[index_p(itheta1, kprow, kpdiag-2)]-Mpkp_p2*aDbD_c[index_p(itheta1, krow, kdiag+2)]);

                // qp = -0
                t5 = 2.0*Aq_m1*Aq_p1*(-Lmk_0*aDa_c[i]+Lmkp_0*aDa_c[i]
                        -Mpk_0*ab_c[ip]-Mpkp_0*aDbD_c[i]);

                // qp = +2 q0
                t6 = pow(Aq_p1,2.0)*(-Lmk_p2*aDa_c[index_p(itheta1, krow+2, kdiag-2)]+Lmkp_m2*aDa_c[index_p(itheta1, krow, kdiag-2)]
                        -Mpk_p2*ab_c[index_p(itheta1, kprow, kpdiag+2)]-Mpkp_m2*aDbD_c[index_p(itheta1, krow, kdiag-2)]);

                aDa_ddt[i] = -I/hbar*(t1+kdotA[itheta1]*(t2+t3)+A0eff2*(t4+t5+t6));

                // fourth equation:

                //first term
                t1 = (R_kp-R_k)*bDb_c[i]+C_kp*aDbD_c[ip]+conj(C_k)*ab_c[i];

                // first sum qp = +/- q0
                // qp = + q0
                t2 = Aq_p1*(Lpk_m1*bDb_c[index_p(itheta1, krow-1, kdiag+1)]-Lpkp_p1*bDb_c[index_p(itheta1, krow, kdiag+1)]
                        -Mmk_m1*ab_c[index_p(itheta1, krow-1, kdiag+1)]-Mmkp_p1*aDbD_c[index_p(itheta1, kprow+1, kpdiag-1)]);

                // qp = - q0
                t3 = Aq_m1*(Lpk_p1*bDb_c[index_p(itheta1, krow+1, kdiag-1)]-Lpkp_m1*bDb_c[index_p(itheta1, krow, kdiag-1)]
                        -Mmk_p1*ab_c[index_p(itheta1, krow+1, kdiag-1)]-Mmkp_m1*aDbD_c[index_p(itheta1, kprow-1, kpdiag+1)]);

                // second sum qp = +/- 2q0, 0
                // qp = -2 q0
                t4 = pow(Aq_m1,2.0)*(-Lmk_p2*bDb_c[index_p(itheta1, krow+2, kdiag-2)]+Lmkp_m2*bDb_c[index_p(itheta1, krow, kdiag-2)]
                        -Mpkp_m2*aDbD_c[index_p(itheta1, kprow-2, kpdiag+2)]-Mpk_p2*ab_c[index_p(itheta1, krow+2, kdiag-2)]);

                // qp = -0
                t5 = 2.0*Aq_m1*Aq_p1*(-Lmk_0*bDb_c[i]+Lmkp_0*bDb_c[i]
                        -Mpkp_0*aDbD_c[ip]-Mpk_0*ab_c[i]);

                // qp = +2 q0
                t6 = pow(Aq_p1,2.0)*(-Lmk_m2*bDb_c[index_p(itheta1, krow-2, kdiag+2)]+Lmkp_p2*bDb_c[index_p(itheta1, krow, kdiag+2)]
                        -Mpkp_p2*aDbD_c[index_p(itheta1, kprow+2, kpdiag-2)]-Mpk_m2*ab_c[index_p(itheta1, krow-2, kdiag+2)]);

                bDb_ddt[i] = -I/hbar*(t1+kdotA[itheta1]*(t2+t3)+A0eff2*(t4+t5+t6));

            }
        }
    }
}

void diffEq2(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[]){
    // input time, matrices indices, matrices for calculations

    // Local variables for the momentum k = ki,  k' = ki + ji entry (ji is one of {-4,-3,-2,-1,0,1,2,3,4}
    // _m2 indicates q' = -2*q0, _m1 is q' = -q0, _0 is q0 = 0; _p1 is q' = q0; _p2 is q' = 2*q0
    // variabels to be changed at each time step (pre-calculated before the calculating the result of the differential equation)
    double epsk_0,
            Ek_0,
            uk_0,
            vk_0,
            R_k, Deltak_0;

    dcomp Delta_ck, C_k, t1;
    int ki, krow, i;

    // Initialize ddt arrays
    //    aDa_ddt = new dcomp[N[itheta1]*Ndiag]();
    //    bDb_ddt = new dcomp[N[itheta1]*Ndiag]();
    //    aDbD_ddt = new dcomp[N[itheta1]*Ndiag]();
    //    ab_ddt = new dcomp[N[itheta1]*Ndiag]();

    //    int kpi = ki+ji;
    // Calculate Aq_p1, Aq_m1

    for (ki = 0; ki<Nk[itheta1]; ki++){
        // Pre-calculate all the variables needed for the differential equation for variable k:
        krow = ki+arrayBuffer;

        // calculate epsk for q' = -2q0, ..., 2q0,
        epsk_0= epsK(ki, itheta1);

        // calculate Delta0(k) for q' = -2q0, ..., 2q0,
        Deltak_0 = calcDelta0k(itheta1,ki);

        // calculate Ek for q' = -2q0, ..., 2q0,
        Ek_0 = Ek(epsk_0, Deltak_0);

        // calculate uk for q' = -2q0, ..., 2q0,
        uk_0 = uk(epsk_0, Ek_0);

        // calculate vk for q' = -2q0, ..., 2q0,
        vk_0 = vk(epsk_0, Ek_0, Deltak_0);

        // Calculate Delta_c(k)
        Delta_ck = calcDeltak(Delta_c, itheta1, ki);

        // calculate Ck, Rk, Ckp, Rkp
        C_k = Ck(epsk_0,uk_0,vk_0,Delta_ck); R_k = Rk(epsk_0,uk_0,vk_0,Delta_ck);

        i = index_diag(itheta1, krow);
        // Pre-calculate all the kp functions

        //
        // Carry out the differential equation RHS for equation 1-4
        //

        // first equation:

        //first term
        t1 = -(R_k+R_k)*aDbD_c[i]+conj(C_k)*aDa_c[i]+conj(C_k)*(bDb_c[i]-dcomp(1.0));

        aDbD_ddt[i] = -I/hbar*t1;

        // third equation:

        //first term
        t1 = C_k*aDbD_c[i]+conj(C_k)*conj(-aDbD_c[i]);

        aDa_ddt[i] = -I/hbar*t1;

        // fourth equation:

        //first term
        t1 = C_k*aDbD_c[i]+conj(C_k)*conj(-aDbD_c[i]);

        bDb_ddt[i] = -I/hbar*t1;

    }
}

void diffEq3(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], dcomp aDa_pr[], dcomp bDb_pr[], dcomp aDbD_pr[], dcomp ab_pr[]){


	
	double epsk_pr, Ek_pr, uk_pr, vk_pr, R_kpr, Lpk_pr, Mmk_pr, Deltak_pr;

	double epsk_0,
	Ek_0,
	uk_0,
	vk_0,
	R_k, Deltak_0;
	
	dcomp Delta_ck, C_k, Delta_ckpr, C_kpr, t1, t2;
	int ki, krow, i_p, i2_p, i_pr;
	
	int idt;
	
	// probe variables
	for (ki = 0; ki<Nk[itheta1]; ki++){

		krow = ki+arrayBuffer;
		// pump indices
		i_p = index_diag(itheta1, krow);
		i2_p = index_diag(itheta1, krow+1);
	
		// Pre-calculate all the variables needed for the differential equation for variable k:
		krow = ki+arrayBuffer;

		// calculate epsk for q' = -2q0, ..., 2q0,
		epsk_0= epsK(ki, itheta1);

		// calculate Delta0(k) for q' = -2q0, ..., 2q0,
		Deltak_0 = calcDelta0k(itheta1,ki);

		// calculate Ek for q' = -2q0, ..., 2q0,
		Ek_0 = Ek(epsk_0, Deltak_0);

		// calculate uk for q' = -2q0, ..., 2q0,
		uk_0 = uk(epsk_0, Ek_0);

		// calculate vk for q' = -2q0, ..., 2q0,
		vk_0 = vk(epsk_0, Ek_0, Deltak_0);

		// Calculate Delta_c(k)
		Delta_ck = calcDeltak(Delta_c, itheta1, ki);

		// calculate Ck, Rk, Ckp, Rkp
		C_k = Ck(epsk_0,uk_0,vk_0,Delta_ck); R_k = Rk(epsk_0,uk_0,vk_0,Delta_ck);
		
		//////////////////// pump ////////////////////

        //
        // Carry out the differential equation RHS for equation 1-4
        //

        // first equation:

        //first term
        t1 = -(R_k+R_k)*aDbD_c[i_p]+conj(C_k)*aDa_c[i_p]+conj(C_k)*(bDb_c[i_p]-dcomp(1.0));

        aDbD_ddt[i_p] = -I/hbar*t1;

        // third equation:

        //first term
        t1 = C_k*aDbD_c[i_p]+conj(C_k)*conj(-aDbD_c[i_p]);

        aDa_ddt[i_p] = -I/hbar*t1;

        // fourth equation:

        //first term
        t1 = C_k*aDbD_c[i_p]+conj(C_k)*conj(-aDbD_c[i_p]);

        bDb_ddt[i_p] = -I/hbar*t1;
        
        /////////////////// PROBE /////////////////////////

		// Probe variables:
		
		Deltak_pr = calcDelta0k(itheta1, ki+1);
		Delta_ckpr = calcDeltak(Delta_c, itheta1, ki+1);

		// calculate epsk for q' = -2q0, ..., 2q0,
		epsk_pr = epsK_pr(ki, itheta1);

		// calculate Ek for q' = -2q0, ..., 2q0,
		Ek_pr = Ek(epsk_pr, Deltak_pr);

		// calculate uk for q' = -2q0, ..., 2q0,
		uk_pr = uk(epsk_pr, Ek_pr);

		// calculate vk for q' = -2q0, ..., 2q0,
		vk_pr = vk(epsk_pr, Ek_pr, Deltak_pr);

		// Calculate Ckp, Rkp
		C_kpr = Ck(epsk_pr,uk_pr,vk_pr,Delta_ckpr); R_kpr = Rk(epsk_pr,uk_pr,vk_pr,Delta_ckpr);
        
        // Calculate L_{k,q'}^{+} for q' = -2q0, ..., 2q0
        Lpk_pr = uk_0*uk_pr+vk_0*vk_pr;
        
        // Calculate M_{k,q'}^{-} for q' = -2q0, ..., 2q0
        Mmk_pr = vk_0*uk_pr-uk_0*vk_pr;

		//
		// Carry out the differential equation RHS for equation 1-4 on the probe variables
		//
		for (idt = 0; idt<Ndt; idt++){
			double dt = calcdt(idt);
								
			if (t>dt-dt_pr && t<dt+dt_pr)
			{		
				i_pr = index_pr(itheta1, idt, krow);
				
				double tminusdt = t-dt;
				dcomp Aq_p1 = Aq_pr(tminusdt);	

				// first equation:
		
				//first term
				t1 = -(R_k+R_kpr)*aDbD_pr[i_pr]+conj(C_kpr)*aDa_pr[i_pr]+conj(C_k)*bDb_pr[i_pr];
		
				// first sum qp = +/- q0
				// qp = + q0
				t2 = Aq_p1*(-Lpk_pr*aDbD_c[i2_p]+Lpk_pr*aDbD_c[i_p]
						+Mmk_pr*aDa_c[i_p]+Mmk_pr*(bDb_c[i2_p]-1.0));
		
				aDbD_ddt_pr[i_pr] = -I/hbar*(t1+kdotA_pr[itheta1]*t2);
		
				// second equation:
		
				//first term
				t1 = (R_k+R_kpr)*ab_pr[i_pr]+C_k*aDa_pr[i_pr]+C_kpr*(bDb_pr[i_pr]);
		
				// first sum qp = +/- q0
				// qp = + q0
				t2 = Aq_p1*(Lpk_pr*conj(-aDbD_c[i_p])-Lpk_pr*conj(-aDbD_c[i2_p])
						-Mmk_pr*aDa_c[i2_p]-Mmk_pr*(bDb_c[i_p]-1.0));
		
				ab_ddt_pr[i_pr] = -I/hbar*(t1+kdotA_pr[itheta1]*t2);
		
				// third equation:
		
				//first term
				t1 = (R_kpr-R_k)*aDa_pr[i_pr]+C_kpr*aDbD_pr[i_pr]+conj(C_k)*ab_pr[i_pr];
		
				// first sum qp = +/- q0
				// qp = + q0
				t2 = Aq_p1*(-Lpk_pr*aDa_c[i2_p]+Lpk_pr*aDa_c[i_p]
						+Mmk_pr*conj(-aDbD_c[i2_p])-Mmk_pr*aDbD_c[i_p]);
		
		
				aDa_ddt_pr[i_pr] = -I/hbar*(t1+kdotA_pr[itheta1]*t2);
		
				// fourth equation:
		
				//first term
				t1 = (R_k-R_kpr)*bDb_pr[i_pr]+C_k*aDbD_pr[i_pr]+conj(C_kpr)*ab_pr[i_pr];
		
				// first sum qp = +/- q0
				// qp = + q0
				t2 = Aq_p1*(Lpk_pr*bDb_c[i_p]-Lpk_pr*bDb_c[i2_p]
						+Mmk_pr*conj(-aDbD_c[i_p])-Mmk_pr*aDbD_c[i2_p]);
		
				bDb_ddt_pr[i_pr] = -I/hbar*(t1+kdotA_pr[itheta1]*t2);
			}
			
			else if (t>=dt+dt_pr) {//first term
				
				i_pr = index_pr(itheta1, idt, krow);
				// if (i_pr>N_pr){
					// printf("Error 1: %d\t%d\t%d\t%f\n", itheta1, idt, krow, t);
					// i_pr = 0;
				// }

				t1 = -(R_k+R_kpr)*aDbD_pr[i_pr]+conj(C_kpr)*aDa_pr[i_pr]+conj(C_k)*bDb_pr[i_pr];
		
				aDbD_ddt_pr[i_pr] = -I/hbar*t1;
		
				// second equation:
		
				//first term
				t1 = (R_k+R_kpr)*ab_pr[i_pr]+C_k*aDa_pr[i_pr]+C_kpr*(bDb_pr[i_pr]);
		
				ab_ddt_pr[i_pr] = -I/hbar*t1;
		
				// third equation:
		
				//first term
				t1 = (R_kpr-R_k)*aDa_pr[i_pr]+C_kpr*aDbD_pr[i_pr]+conj(C_k)*ab_pr[i_pr];
		
				aDa_ddt_pr[i_pr] = -I/hbar*t1;
		
				// fourth equation:
		
				//first term
				t1 = (R_k-R_kpr)*bDb_pr[i_pr]+C_k*aDbD_pr[i_pr]+conj(C_kpr)*ab_pr[i_pr];
		
				bDb_ddt_pr[i_pr] = -I/hbar*t1;
			}
		}

    }
}

void diffEq4(int itheta1, double t, dcomp Delta_c, dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], dcomp aDa_pr[], dcomp bDb_pr[], dcomp aDbD_pr[], dcomp ab_pr[]){

	double epsk_pr, Ek_pr, uk_pr, vk_pr, R_kpr, Deltak_pr;

	double epsk_0,
	Ek_0,
	uk_0,
	vk_0,
	R_k, Deltak_0, dt;
	
	dcomp Delta_ck, C_k, Delta_ckpr, C_kpr, t1;
	int ki, krow, i_p, i_pr, idt;
	

	for (ki = 0; ki<Nk[itheta1]; ki++){

		krow = ki+arrayBuffer;
		i_p = index_diag(itheta1, krow);

		// Pre-calculate all the variables needed for the differential equation for variable k:

		// calculate epsk for q' = -2q0, ..., 2q0,
		epsk_0= epsK(ki, itheta1);

		// calculate Delta0(k) for q' = -2q0, ..., 2q0,
		Deltak_0 = calcDelta0k(itheta1,ki);
		
		// calculate Ek for q' = -2q0, ..., 2q0,
		Ek_0 = Ek(epsk_0, Deltak_0);

		// calculate uk for q' = -2q0, ..., 2q0,
		uk_0 = uk(epsk_0, Ek_0);

		// calculate vk for q' = -2q0, ..., 2q0,
		vk_0 = vk(epsk_0, Ek_0, Deltak_0);

		// Calculate Delta_c(k)
		Delta_ck = calcDeltak(Delta_c, itheta1, ki);

		// calculate Ck, Rk, Ckp, Rkp
		C_k = Ck(epsk_0,uk_0,vk_0,Delta_ck); R_k = Rk(epsk_0,uk_0,vk_0,Delta_ck);
		
		//////////////////// pump ////////////////////

        //
        // Carry out the differential equation RHS for equation 1-4
        //

        // first equation:

        //first term
        t1 = -(R_k+R_k)*aDbD_c[i_p]+conj(C_k)*aDa_c[i_p]+conj(C_k)*(bDb_c[i_p]-dcomp(1.0));

        aDbD_ddt[i_p] = -I/hbar*t1;

        // third equation:

        //first term
        t1 = C_k*aDbD_c[i_p]+conj(C_k)*conj(-aDbD_c[i_p]);

        aDa_ddt[i_p] = -I/hbar*t1;

        // fourth equation:

        //first term
        t1 = C_k*aDbD_c[i_p]+conj(C_k)*conj(-aDbD_c[i_p]);

        bDb_ddt[i_p] = -I/hbar*t1;
		
		////////////////// PROBE ////////////////////
		// Probe variables:
		
		// calculate Delta0(k) for q' = -2q0, ..., 2q0,
		Deltak_pr = calcDelta0k(itheta1,ki+1);
		Delta_ckpr = calcDeltak(Delta_c,itheta1,ki+1);

		// calculate epsk for q' = -2q0, ..., 2q0,
		epsk_pr = epsK_pr(ki, itheta1);

		// calculate Ek for q' = -2q0, ..., 2q0,
		Ek_pr = Ek(epsk_pr, Deltak_pr);

		// calculate uk for q' = -2q0, ..., 2q0,
		uk_pr = uk(epsk_pr, Ek_pr);

		// calculate vk for q' = -2q0, ..., 2q0,
		vk_pr = vk(epsk_pr, Ek_pr, Deltak_pr);

		// Calculate Ckp, Rkp
		C_kpr = Ck(epsk_pr,uk_pr,vk_pr,Delta_ckpr); R_kpr = Rk(epsk_pr,uk_pr,vk_pr,Delta_ckpr);


		//
		// Carry out the differential equation RHS for equation 1-4 on the probe variables
		//
		for (idt = 0; idt<Ndt; idt++){
			dt = calcdt(idt);
			if(t<tend+dt){			
				i_pr = index_pr(itheta1, idt, krow);

				// first equation:
		
				//first term
				t1 = -(R_k+R_kpr)*aDbD_pr[i_pr]+conj(C_kpr)*aDa_pr[i_pr]+conj(C_k)*bDb_pr[i_pr];
		
				aDbD_ddt_pr[i_pr] = -I/hbar*t1;
		
				// second equation:
		
				//first term
				t1 = (R_k+R_kpr)*ab_pr[i_pr]+C_k*aDa_pr[i_pr]+C_kpr*(bDb_pr[i_pr]);
		
				ab_ddt_pr[i_pr] = -I/hbar*t1;
		
				// third equation:
		
				//first term
				t1 = (R_kpr-R_k)*aDa_pr[i_pr]+C_kpr*aDbD_pr[i_pr]+conj(C_k)*ab_pr[i_pr];
		
				aDa_ddt_pr[i_pr] = -I/hbar*t1;
		
				// fourth equation:
		
				//first term
				t1 = (R_k-R_kpr)*bDb_pr[i_pr]+C_k*aDbD_pr[i_pr]+conj(C_kpr)*ab_pr[i_pr];
		
				bDb_ddt_pr[i_pr] = -I/hbar*t1;
			}
		}
    }
}

void RungeKutta(double t, double h){
    int itheta1, ki, ji, kdiag, krow;
    dcomp Delta_c;                                                    // meV                  the complex-valued gap at the current time

    // dynamically modify loop size for w_d dependence. call this n_c ??
    // also need to change for W and Delta
#pragma omp parallel private(ki, ji, kdiag, krow)
    {


#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // First Step:

            // uses the _k arrays
            // store the value in _ddt arrays: k_1
            diffEq(itheta1, t, Delta, aDa, bDb, aDbD, ab);

            // Calculate the current time derivatives: f(x_n,y_n)
            for (ji = -4; ji < 5; ji++){
                for (ki = 0; ki<Nk[itheta1]; ki++){
                    if (ji + ki>=0 && ji + ki < Nk[itheta1]){                                                // check every time that we are within the buffer

                        krow = ki+arrayBuffer;

                        kdiag = ji+6;

                        // calculates the aggregate value _rk: y_n + k_1 / 6
                        aDa_rk[index_p(itheta1,krow,kdiag)] = aDa[index_p(itheta1,krow,kdiag)] + h*aDa_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                        bDb_rk[index_p(itheta1,krow,kdiag)] = bDb[index_p(itheta1,krow,kdiag)] + h*bDb_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                        aDbD_rk[index_p(itheta1,krow,kdiag)] = aDbD[index_p(itheta1,krow,kdiag)] + h*aDbD_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                        ab_rk[index_p(itheta1,krow,kdiag)] = ab[index_p(itheta1,krow,kdiag)] + h*ab_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                    }
                }
            }

            for (ji = -4; ji < 5; ji++){
                for (ki = 0; ki<Nk[itheta1]; ki++){
                    if (ji + ki>=0 && ji + ki < Nk[itheta1]){                                                // check every time that we are within the buffer
                        krow = ki+arrayBuffer;
                        kdiag = ji+6;
                        // calculates the value for the next iteration _k: y_n + k_1 / 2 at t = t_i + h / 2
                        aDa_k[index_p(itheta1,krow,kdiag)] = aDa[index_p(itheta1,krow,kdiag)] + 0.5*h*aDa_ddt[index_p(itheta1, krow, kdiag)];
                        bDb_k[index_p(itheta1,krow,kdiag)] = bDb[index_p(itheta1,krow,kdiag)] + 0.5*h*bDb_ddt[index_p(itheta1, krow, kdiag)];
                        aDbD_k[index_p(itheta1,krow,kdiag)] = aDbD[index_p(itheta1,krow,kdiag)] + 0.5*h*aDbD_ddt[index_p(itheta1, krow, kdiag)];
                        ab_k[index_p(itheta1,krow,kdiag)] = ab[index_p(itheta1,krow,kdiag)] + 0.5*h*ab_ddt[index_p(itheta1, krow, kdiag)];
                    }
                }
            }
            // Delete ddt arrays
            //        delete[] aDa_ddt;
            //        delete[] bDb_ddt;
            //        delete[] aDbD_ddt;
            //        delete[] ab_ddt;
        }
#pragma omp single
        {
            Delta_c = gapEq(aDa_k,bDb_k,aDbD_k,ab_k);
            t = t + 0.5*h; // update the time step for k_2
        }

        // Second Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_2
            diffEq(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, ab_k);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_1/2)
            for (ji = -4; ji < 5; ji++){
                for (ki = 0; ki<Nk[itheta1]; ki++){
                    if (ji + ki>=0 && ji + ki < Nk[itheta1]){                                                // check every time that we are within the buffer
                        krow = ki+arrayBuffer;
                        kdiag = ji+6;
                        // calculates the aggregate value _rk: y_n + k_1 / 3
                        aDa_rk[index_p(itheta1,krow,kdiag)] = aDa_rk[index_p(itheta1,krow,kdiag)] + h*aDa_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                        bDb_rk[index_p(itheta1,krow,kdiag)] = bDb_rk[index_p(itheta1,krow,kdiag)] + h*bDb_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                        aDbD_rk[index_p(itheta1,krow,kdiag)] = aDbD_rk[index_p(itheta1,krow,kdiag)] + h*aDbD_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                        ab_rk[index_p(itheta1,krow,kdiag)] = ab_rk[index_p(itheta1,krow,kdiag)] + h*ab_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                    }
                }
            }

            for (ji = -4; ji < 5; ji++){
                for (ki = 0; ki<Nk[itheta1]; ki++){
                    if (ji + ki>=0 && ji + ki < Nk[itheta1]){                                                // check every time that we are within the buffer
                        krow = ki+arrayBuffer;
                        kdiag = ji+6;
                        // calculates the value for the next iteration _k: y_n + k_2 / 2 at t = t_i + h / 2
                        aDa_k[index_p(itheta1,krow,kdiag)] = aDa[index_p(itheta1,krow,kdiag)] + 0.5*h*aDa_ddt[index_p(itheta1, krow, kdiag)];
                        bDb_k[index_p(itheta1,krow,kdiag)] = bDb[index_p(itheta1,krow,kdiag)] + 0.5*h*bDb_ddt[index_p(itheta1, krow, kdiag)];
                        aDbD_k[index_p(itheta1,krow,kdiag)] = aDbD[index_p(itheta1,krow,kdiag)] + 0.5*h*aDbD_ddt[index_p(itheta1, krow, kdiag)];
                        ab_k[index_p(itheta1,krow,kdiag)] = ab[index_p(itheta1,krow,kdiag)] + 0.5*h*ab_ddt[index_p(itheta1, krow, kdiag)];
                    }
                }
            }
            // Delete ddt arrays
            //        delete[] aDa_ddt;
            //        delete[] bDb_ddt;
            //        delete[] aDbD_ddt;
            //        delete[] ab_ddt;
        }
#pragma omp single
        {
            Delta_c = gapEq(aDa_k, bDb_k, aDbD_k, ab_k);
        }

        // Third Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_3
            diffEq(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, ab_k);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_2 / 2)
            for (ji = -4; ji < 5; ji++){
                for (ki = 0; ki<Nk[itheta1]; ki++){
                    if (ji + ki>=0 && ji + ki < Nk[itheta1]){                                                // check every time that we are within the buffer
                        krow = ki+arrayBuffer;
                        kdiag = ji+6;

                        // calculates the aggregate value _rk: y_n + k_1 / 3
                        aDa_rk[index_p(itheta1,krow,kdiag)] = aDa_rk[index_p(itheta1,krow,kdiag)] + h*aDa_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                        bDb_rk[index_p(itheta1,krow,kdiag)] = bDb_rk[index_p(itheta1,krow,kdiag)] + h*bDb_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                        aDbD_rk[index_p(itheta1,krow,kdiag)] = aDbD_rk[index_p(itheta1,krow,kdiag)] + h*aDbD_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                        ab_rk[index_p(itheta1,krow,kdiag)] = ab_rk[index_p(itheta1,krow,kdiag)] + h*ab_ddt[index_p(itheta1, krow, kdiag)] / 3.0;
                    }
                }
            }

            for (ji = -4; ji < 5; ji++){
                for (ki = 0; ki<Nk[itheta1]; ki++){
                    if (ji + ki>=0 && ji + ki < Nk[itheta1]){                                                // check every time that we are within the buffer
                        krow = ki+arrayBuffer;
                        kdiag = ji+6;
                        // calculates the value for the next iteration _k: y_n + k_3 at t = t_i + h
                        aDa_k[index_p(itheta1,krow,kdiag)] = aDa[index_p(itheta1,krow,kdiag)] + h*aDa_ddt[index_p(itheta1, krow, kdiag)];
                        bDb_k[index_p(itheta1,krow,kdiag)] = bDb[index_p(itheta1,krow,kdiag)] + h*bDb_ddt[index_p(itheta1, krow, kdiag)];
                        aDbD_k[index_p(itheta1,krow,kdiag)] = aDbD[index_p(itheta1,krow,kdiag)] + h*aDbD_ddt[index_p(itheta1, krow, kdiag)];
                        ab_k[index_p(itheta1,krow,kdiag)] = ab[index_p(itheta1,krow,kdiag)] + h*ab_ddt[index_p(itheta1, krow, kdiag)];
                    }
                }
            }
            // Delete ddt arrays
            //        delete[] aDa_ddt;
            //        delete[] bDb_ddt;
            //        delete[] aDbD_ddt;
            //        delete[] ab_ddt;
        }
#pragma omp single
        {
            Delta_c = gapEq(aDa_k, bDb_k, aDbD_k, ab_k);
			t=t+0.5*h;
        }
        // Fourth Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_4
            diffEq(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, ab_k);

            // Calculate the current time derivatives: f(x_n + h, y_n + k_3)
            for (ji = -4; ji < 5; ji++){
                for (ki = 0; ki<Nk[itheta1]; ki++){
                    if (ji + ki>=0 && ji + ki < Nk[itheta1]){                                                // check every time that we are within the buffer
                        krow = ki+arrayBuffer;
                        kdiag = ji+6;

                        // calculates the aggregate value: y_n + k_4 / 6 and stores it in the main matrices as the new y_{n+1}
                        aDa[index_p(itheta1,krow,kdiag)] = aDa_rk[index_p(itheta1,krow,kdiag)] + h*aDa_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                        bDb[index_p(itheta1,krow,kdiag)] = bDb_rk[index_p(itheta1,krow,kdiag)] + h*bDb_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                        aDbD[index_p(itheta1,krow,kdiag)] = aDbD_rk[index_p(itheta1,krow,kdiag)] + h*aDbD_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                        ab[index_p(itheta1,krow,kdiag)] = ab_rk[index_p(itheta1,krow,kdiag)] + h*ab_ddt[index_p(itheta1, krow, kdiag)] / 6.0;
                    }
                }
            }
            // Delete ddt arrays
            //        delete[] aDa_ddt;
            //        delete[] bDb_ddt;
            //        delete[] aDbD_ddt;
            //        delete[] ab_ddt;
        }
    } // end of parallel region
    Delta = gapEq(aDa, bDb, aDbD, ab);
}

void RungeKutta2(double t, double h){
    int itheta1, ki, krow, i;
    dcomp Delta_c;                                                    // meV                  the complex-valued gap at the current time

    FILE * data_Delta;
    data_Delta = fopen("data_Delta.txt","a");

    // dynamically modify loop size for w_d dependence. call this n_c ??
    // also need to change for W and Delta
#pragma omp parallel private(i, ki, krow)
    {


#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // First Step:

            // uses the _k arrays
            // store the value in _ddt arrays: k_1
            diffEq2(itheta1, t, Delta, aDa, bDb, aDbD);

            // Calculate the current time derivatives: f(x_n,y_n)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 6
                aDa_rk[i] = aDa[i] + h*aDa_ddt[i] / 6.0;
                bDb_rk[i] = bDb[i] + h*bDb_ddt[i] / 6.0;
                aDbD_rk[i] = aDbD[i] + h*aDbD_ddt[i] / 6.0;
            }




            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_1 / 2 at t = t_i + h / 2
                aDa_k[i] = aDa[i] + 0.5*h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + 0.5*h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + 0.5*h*aDbD_ddt[i];
            }
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k,bDb_k,aDbD_k);
            t = t + 0.5*h; // update the time step for k_2
            
            fprintf(data_Delta, "%e\t", real(Delta));
            fprintf(data_Delta, "%e\n", imag(Delta));
        }

        // Second Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_2
            diffEq2(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_1/2)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 3
                aDa_rk[i] = aDa_rk[i] + h*aDa_ddt[i] / 3.0;
                bDb_rk[i] = bDb_rk[i] + h*bDb_ddt[i] / 3.0;
                aDbD_rk[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 3.0;
            }


            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_2 / 2 at t = t_i + h / 2
                aDa_k[i] = aDa[i] + 0.5*h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + 0.5*h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + 0.5*h*aDbD_ddt[i];
            }
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k, bDb_k, aDbD_k);
            
            fprintf(data_Delta, "%e\t", real(Delta));
            fprintf(data_Delta, "%e\n", imag(Delta));
        }

        // Third Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_3
            diffEq2(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_2 / 2)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 3
                aDa_rk[i] = aDa_rk[i] + h*aDa_ddt[i] / 3.0;
                bDb_rk[i] = bDb_rk[i] + h*bDb_ddt[i] / 3.0;
                aDbD_rk[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 3.0;
            }

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_3 at t = t_i + h
                aDa_k[i] = aDa[i] + h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + h*aDbD_ddt[i];
            }
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k, bDb_k, aDbD_k);
			t=t+0.5*h;
            
            fprintf(data_Delta, "%e\t", real(Delta));
            fprintf(data_Delta, "%e\n", imag(Delta));
        }
        // Fourth Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_4
            diffEq2(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k);

            // Calculate the current time derivatives: f(x_n + h, y_n + k_3)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);

                // calculates the aggregate value: y_n + k_4 / 6 and stores it in the main matrices as the new y_{n+1}
                aDa[i] = aDa_rk[i] + h*aDa_ddt[i] / 6.0;
                bDb[i] = bDb_rk[i] + h*bDb_ddt[i] / 6.0;
                aDbD[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 6.0;
            }
        }
    } // end of parallel region
    Delta = gapEq2(aDa, bDb, aDbD);
    
    fprintf(data_Delta, "%e\t", real(Delta));
    fprintf(data_Delta, "%e\n", imag(Delta));
    fclose(data_Delta);
}

void RungeKutta3(double t, double h){
    int itheta1, ki, krow, i, idt;
    dcomp Delta_c;                                                    // meV                  the complex-valued gap at the current time

    // dynamically modify loop size for w_d dependence. call this n_c ??
    // also need to change for W and Delta
#pragma omp parallel private(i, ki, krow, idt)
    {


#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // First Step:

            // uses the _k arrays
            // store the value in _ddt arrays: k_1
            diffEq3(itheta1, t, Delta, aDa, bDb, aDbD, aDa_pr, bDb_pr, aDbD_pr, ab_pr);

            // Calculate the current time derivatives: f(x_n,y_n)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 6
                aDa_rk[i] = aDa[i] + h*aDa_ddt[i] / 6.0;
                bDb_rk[i] = bDb[i] + h*bDb_ddt[i] / 6.0;
                aDbD_rk[i] = aDbD[i] + h*aDbD_ddt[i] / 6.0;
            }

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_1 / 2 at t = t_i + h / 2
                aDa_k[i] = aDa[i] + 0.5*h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + 0.5*h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + 0.5*h*aDbD_ddt[i];
            }
            for (idt = 0; idt<Ndt; idt++){
				double dt = calcdt(idt);
				if (t>dt-dt_pr){
					for (ki = 0; ki<Nk[itheta1]; ki++){
					
						i = index_pr(itheta1, idt, ki+arrayBuffer);
						
						aDa_rk_pr[i] = aDa_pr[i] + h*aDa_ddt_pr[i] / 6.0;
						bDb_rk_pr[i] = bDb_pr[i] + h*bDb_ddt_pr[i] / 6.0;
						aDbD_rk_pr[i] = aDbD_pr[i] + h*aDbD_ddt_pr[i] / 6.0;
						ab_rk_pr[i] = ab_pr[i] + h*ab_ddt_pr[i] / 6.0;
						
					}
				}
			}
			
			for (idt = 0; idt<Ndt; idt++) {
				double dt = calcdt(idt);
				if (t>dt-dt_pr){
				for (ki = 0; ki<Nk[itheta1]; ki++){
						
						i = index_pr(itheta1, idt, ki+arrayBuffer);
						
						aDa_k_pr[i] = aDa_pr[i] + 0.5*h*aDa_ddt_pr[i];
						bDb_k_pr[i] = bDb_pr[i] + 0.5*h*bDb_ddt_pr[i];
						aDbD_k_pr[i] = aDbD_pr[i] + 0.5*h*aDbD_ddt_pr[i];
						ab_k_pr[i] = ab_pr[i] + 0.5*h*ab_ddt_pr[i];
					}
				}
			}
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k,bDb_k,aDbD_k);
            t = t + 0.5*h; // update the time step for k_2
        }

        // Second Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_2
            diffEq3(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, aDa_k_pr, bDb_k_pr, aDbD_k_pr, ab_k_pr);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_1/2)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 3
                aDa_rk[i] = aDa_rk[i] + h*aDa_ddt[i] / 3.0;
                bDb_rk[i] = bDb_rk[i] + h*bDb_ddt[i] / 3.0;
                aDbD_rk[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 3.0;
            }


            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_2 / 2 at t = t_i + h / 2
                aDa_k[i] = aDa[i] + 0.5*h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + 0.5*h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + 0.5*h*aDbD_ddt[i];
            }
            
            for (idt = 0; idt<Ndt; idt++){
				double dt = calcdt(idt);
				if (t>dt-dt_pr){				
					for (ki = 0; ki<Nk[itheta1]; ki++){
			
						i = index_pr(itheta1, idt, ki+arrayBuffer);
			
						aDa_rk_pr[i] = aDa_rk_pr[i] + h*aDa_ddt_pr[i] / 3.0;
						bDb_rk_pr[i] = bDb_rk_pr[i] + h*bDb_ddt_pr[i] / 3.0;
						aDbD_rk_pr[i] = aDbD_rk_pr[i] + h*aDbD_ddt_pr[i] / 3.0;
						ab_rk_pr[i] = ab_rk_pr[i] + h*ab_ddt_pr[i] / 3.0;
					}
				}
			}
			
			for (idt=0; idt<Ndt; idt++){
				double dt = calcdt(idt);
				if (t>dt-dt_pr){
					for (ki = 0; ki<Nk[itheta1]; ki++){
		
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_k_pr[i] = aDa_pr[i] + 0.5*h*aDa_ddt_pr[i];
						bDb_k_pr[i] = bDb_pr[i] + 0.5*h*bDb_ddt_pr[i];
						aDbD_k_pr[i] = aDbD_pr[i] + 0.5*h*aDbD_ddt_pr[i];
						ab_k_pr[i] = ab_pr[i] + 0.5*h*ab_ddt_pr[i];
					}
				}
			}
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k, bDb_k, aDbD_k);
        }

        // Third Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_3
            diffEq3(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, aDa_k_pr, bDb_k_pr, aDbD_k_pr, ab_k_pr);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_2 / 2)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 3
                aDa_rk[i] = aDa_rk[i] + h*aDa_ddt[i] / 3.0;
                bDb_rk[i] = bDb_rk[i] + h*bDb_ddt[i] / 3.0;
                aDbD_rk[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 3.0;
            }

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_3 at t = t_i + h
                aDa_k[i] = aDa[i] + h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + h*aDbD_ddt[i];
            }
            
            
			for (idt=0; idt<Ndt; idt++){
				double dt = calcdt(idt);
				if (t>dt-dt_pr){				
					for (ki = 0; ki<Nk[itheta1]; ki++){
		
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_rk_pr[i] = aDa_rk_pr[i] + h*aDa_ddt_pr[i] / 3.0;
						bDb_rk_pr[i] = bDb_rk_pr[i] + h*bDb_ddt_pr[i] / 3.0;
						aDbD_rk_pr[i] = aDbD_rk_pr[i] + h*aDbD_ddt_pr[i] / 3.0;
						ab_rk_pr[i] = ab_rk_pr[i] + h*ab_ddt_pr[i] / 3.0;
					}
				}
			}

			for (idt=0; idt<Ndt; idt++){
				double dt = calcdt(idt);
				if (t>dt-dt_pr){				
					for (ki = 0; ki<Nk[itheta1]; ki++){
		
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_k_pr[i] = aDa_pr[i] + h*aDa_ddt_pr[i];
						bDb_k_pr[i] = bDb_pr[i] + h*bDb_ddt_pr[i];
						aDbD_k_pr[i] = aDbD_pr[i] + h*aDbD_ddt_pr[i];
						ab_k_pr[i] = ab_pr[i] + h*ab_ddt_pr[i];
					}
				}
			}
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k, bDb_k, aDbD_k);
			t=t+0.5*h;
        }
        // Fourth Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_4
            diffEq3(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, aDa_k_pr, bDb_k_pr, aDbD_k_pr, ab_k_pr);

            // Calculate the current time derivatives: f(x_n + h, y_n + k_3)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);

                // calculates the aggregate value: y_n + k_4 / 6 and stores it in the main matrices as the new y_{n+1}
                aDa[i] = aDa_rk[i] + h*aDa_ddt[i] / 6.0;
                bDb[i] = bDb_rk[i] + h*bDb_ddt[i] / 6.0;
                aDbD[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 6.0;
            }
            
            for (idt=0; idt<Ndt; idt++){
				double dt = calcdt(idt);
				if (t>dt-dt_pr){				
					for (ki = 0; ki<Nk[itheta1]; ki++){
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_pr[i] = aDa_rk_pr[i] + h*aDa_ddt_pr[i] / 6.0;
						bDb_pr[i] = bDb_rk_pr[i] + h*bDb_ddt_pr[i] / 6.0;
						aDbD_pr[i] = aDbD_rk_pr[i] + h*aDbD_ddt_pr[i] / 6.0;
						ab_pr[i] = ab_rk_pr[i] + h*ab_ddt_pr[i] / 6.0;
					}
				}
			}            
        }
    } // end of parallel region
    Delta = gapEq2(aDa, bDb, aDbD);
}

void RungeKutta4(double t, double h){
    int itheta1, ki, krow, i, idt;
    double dt;
    dcomp Delta_c;                                                    // meV                  the complex-valued gap at the current time

    // dynamically modify loop size for w_d dependence. call this n_c ??
    // also need to change for W and Delta
#pragma omp parallel private(i, ki, krow, idt, dt)
    {


#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // First Step:

            // uses the _k arrays
            // store the value in _ddt arrays: k_1
            diffEq4(itheta1, t, Delta, aDa, bDb, aDbD, aDa_pr, bDb_pr, aDbD_pr, ab_pr);

            // Calculate the current time derivatives: f(x_n,y_n)
            
			for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 6
                aDa_rk[i] = aDa[i] + h*aDa_ddt[i] / 6.0;
                bDb_rk[i] = bDb[i] + h*bDb_ddt[i] / 6.0;
                aDbD_rk[i] = aDbD[i] + h*aDbD_ddt[i] / 6.0;
            }

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_1 / 2 at t = t_i + h / 2
                aDa_k[i] = aDa[i] + 0.5*h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + 0.5*h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + 0.5*h*aDbD_ddt[i];
            }

            
            for (idt = 0; idt<Ndt; idt++){
				dt = calcdt(idt);
				if(t<tend+dt){				
					for (ki = 0; ki<Nk[itheta1]; ki++){
					
						i = index_pr(itheta1, idt, ki+arrayBuffer);
						
						aDa_rk_pr[i] = aDa_pr[i] + h*aDa_ddt_pr[i] / 6.0;
						bDb_rk_pr[i] = bDb_pr[i] + h*bDb_ddt_pr[i] / 6.0;
						aDbD_rk_pr[i] = aDbD_pr[i] + h*aDbD_ddt_pr[i] / 6.0;
						ab_rk_pr[i] = ab_pr[i] + h*ab_ddt_pr[i] / 6.0;
						
					}
				}
			}
			
			for (idt = 0; idt<Ndt; idt++) {
				dt = calcdt(idt);
				if(t<tend+dt){
					for (ki = 0; ki<Nk[itheta1]; ki++){
						
						i = index_pr(itheta1, idt, ki+arrayBuffer);
						
						aDa_k_pr[i] = aDa_pr[i] + 0.5*h*aDa_ddt_pr[i];
						bDb_k_pr[i] = bDb_pr[i] + 0.5*h*bDb_ddt_pr[i];
						aDbD_k_pr[i] = aDbD_pr[i] + 0.5*h*aDbD_ddt_pr[i];
						ab_k_pr[i] = ab_pr[i] + 0.5*h*ab_ddt_pr[i];
					}
				}
			}
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k,bDb_k,aDbD_k);
            t = t + 0.5*h; // update the time step for k_2
        }

        // Second Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_2
            diffEq4(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, aDa_k_pr, bDb_k_pr, aDbD_k_pr, ab_k_pr);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_1/2)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 3
                aDa_rk[i] = aDa_rk[i] + h*aDa_ddt[i] / 3.0;
                bDb_rk[i] = bDb_rk[i] + h*bDb_ddt[i] / 3.0;
                aDbD_rk[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 3.0;
            }


            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_2 / 2 at t = t_i + h / 2
                aDa_k[i] = aDa[i] + 0.5*h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + 0.5*h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + 0.5*h*aDbD_ddt[i];
            }
            
            for (idt = 0; idt<Ndt; idt++){
				dt = calcdt(idt);
				if(t<tend+dt){
					for (ki = 0; ki<Nk[itheta1]; ki++){
			
						i = index_pr(itheta1, idt, ki+arrayBuffer);
			
						aDa_rk_pr[i] = aDa_rk_pr[i] + h*aDa_ddt_pr[i] / 3.0;
						bDb_rk_pr[i] = bDb_rk_pr[i] + h*bDb_ddt_pr[i] / 3.0;
						aDbD_rk_pr[i] = aDbD_rk_pr[i] + h*aDbD_ddt_pr[i] / 3.0;
						ab_rk_pr[i] = ab_rk_pr[i] + h*ab_ddt_pr[i] / 3.0;
					}
				}
			}
			
			for (idt=0; idt<Ndt; idt++){
				dt = calcdt(idt);
				if(t<tend+dt){				
					for (ki = 0; ki<Nk[itheta1]; ki++){
		
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_k_pr[i] = aDa_pr[i] + 0.5*h*aDa_ddt_pr[i];
						bDb_k_pr[i] = bDb_pr[i] + 0.5*h*bDb_ddt_pr[i];
						aDbD_k_pr[i] = aDbD_pr[i] + 0.5*h*aDbD_ddt_pr[i];
						ab_k_pr[i] = ab_pr[i] + 0.5*h*ab_ddt_pr[i];
					}
				}
			}
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k, bDb_k, aDbD_k);
        }

        // Third Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_3
            diffEq4(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, aDa_k_pr, bDb_k_pr, aDbD_k_pr, ab_k_pr);

            // Calculate the current time derivatives: f(x_n + h/2, y_n + k_2 / 2)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the aggregate value _rk: y_n + k_1 / 3
                aDa_rk[i] = aDa_rk[i] + h*aDa_ddt[i] / 3.0;
                bDb_rk[i] = bDb_rk[i] + h*bDb_ddt[i] / 3.0;
                aDbD_rk[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 3.0;
            }

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);
                // calculates the value for the next iteration _k: y_n + k_3 at t = t_i + h
                aDa_k[i] = aDa[i] + h*aDa_ddt[i];
                bDb_k[i] = bDb[i] + h*bDb_ddt[i];
                aDbD_k[i] = aDbD[i] + h*aDbD_ddt[i];
            }
            
            
			for (idt=0; idt<Ndt; idt++){
				dt = calcdt(idt);
				if(t<tend+dt){
					for (ki = 0; ki<Nk[itheta1]; ki++){
		
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_rk_pr[i] = aDa_rk_pr[i] + h*aDa_ddt_pr[i] / 3.0;
						bDb_rk_pr[i] = bDb_rk_pr[i] + h*bDb_ddt_pr[i] / 3.0;
						aDbD_rk_pr[i] = aDbD_rk_pr[i] + h*aDbD_ddt_pr[i] / 3.0;
						ab_rk_pr[i] = ab_rk_pr[i] + h*ab_ddt_pr[i] / 3.0;
					}
				}
			}

			for (idt=0; idt<Ndt; idt++){
				dt = calcdt(idt);
				if(t<tend+dt){
					for (ki = 0; ki<Nk[itheta1]; ki++){
		
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_k_pr[i] = aDa_pr[i] + h*aDa_ddt_pr[i];
						bDb_k_pr[i] = bDb_pr[i] + h*bDb_ddt_pr[i];
						aDbD_k_pr[i] = aDbD_pr[i] + h*aDbD_ddt_pr[i];
						ab_k_pr[i] = ab_pr[i] + h*ab_ddt_pr[i];
					}
				}
			}
        }
#pragma omp single
        {
            Delta_c = gapEq2(aDa_k, bDb_k, aDbD_k);
			t=t+0.5*h;
        }
        // Fourth Step:
#pragma omp for schedule(dynamic, 1)
        for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
        {

            // uses the _k arrays
            // store the value in _ddt arrays: k_4
            diffEq4(itheta1, t, Delta_c, aDa_k, bDb_k, aDbD_k, aDa_k_pr, bDb_k_pr, aDbD_k_pr, ab_k_pr);

            // Calculate the current time derivatives: f(x_n + h, y_n + k_3)

            for (ki = 0; ki<Nk[itheta1]; ki++){

                krow = ki+arrayBuffer;
                i = index_diag(itheta1, krow);

                // calculates the aggregate value: y_n + k_4 / 6 and stores it in the main matrices as the new y_{n+1}
                aDa[i] = aDa_rk[i] + h*aDa_ddt[i] / 6.0;
                bDb[i] = bDb_rk[i] + h*bDb_ddt[i] / 6.0;
                aDbD[i] = aDbD_rk[i] + h*aDbD_ddt[i] / 6.0;
            }
            
            for (idt=0; idt<Ndt; idt++){
				dt = calcdt(idt);
				if(t<tend+dt){
					for (ki = 0; ki<Nk[itheta1]; ki++){
						i = index_pr(itheta1, idt, ki+arrayBuffer);
		
						aDa_pr[i] = aDa_rk_pr[i] + h*aDa_ddt_pr[i] / 6.0;
						bDb_pr[i] = bDb_rk_pr[i] + h*bDb_ddt_pr[i] / 6.0;
						aDbD_pr[i] = aDbD_rk_pr[i] + h*aDbD_ddt_pr[i] / 6.0;
						ab_pr[i] = ab_rk_pr[i] + h*ab_ddt_pr[i] / 6.0;
					}
				}
			}            
        }
    } // end of parallel region
    Delta = gapEq2(aDa, bDb, aDbD);
}

dcomp gapEq(dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[], dcomp ab_c[]){

    dcomp sumDelta, sumDelta2 = 0.0;
    int itheta1p, i;

    sumDelta = 0.0;
// #pragma omp parallel for schedule(dynamic, 1) reduction(+:sumDelta){
    for (itheta1p = 0; itheta1p< Ntheta1; itheta1p++)
    { // start of parallel region
        int ki, krow;
        double uk_0, vk_0, epsk_0, Ek_0, theta1p, Deltak_0;
        dcomp sumDelta1 = 0.0;
        
        for (ki = 0; ki<Nk[itheta1p]; ki++){ // kp along the 2 w_D wide line
            krow = ki+arrayBuffer;
            
            // Calculate the Bogoliubov variables
            epsk_0 = epsK(ki, itheta1p);
            Deltak_0 = calcDelta0k(itheta1p, ki);
            theta1p = calcAnglek(itheta1p, ki);
            Ek_0 = Ek(epsk_0, Deltak_0);
            uk_0 = uk(epsk_0, Ek_0);
            vk_0 = vk(epsk_0, Ek_0, Deltak_0);
            i = index_p(itheta1p, krow, 6);
            
            // Add the k-th term to the sum
            sumDelta1 += dwave(theta1p)*(uk_0*vk_0*(aDa_c[i]+bDb_c[i]-dcomp(1.0))-pow(uk_0,2.0)*ab_c[i]-pow(vk_0,2.0)*aDbD_c[i]);
            
        }

        sumDelta += sumDelta1/dcomp(Nk[itheta1p]);

    } // end of parallel region

return W*sumDelta/dcomp(Ntheta1);
}

dcomp gapEq2(dcomp aDa_c[], dcomp bDb_c[], dcomp aDbD_c[]){
    
    dcomp sumDelta, sumDelta2 = 0.0;
    int itheta1p;
    sumDelta = 0.0;
// #pragma omp parallel for schedule(dynamic, 1) reduction(+:sumDelta)
    for (itheta1p = 0; itheta1p< Ntheta1; itheta1p++)
    { // start of parallel region
        int ki, krow, i;
        double uk_0, vk_0, epsk_0, Ek_0, theta1p, Deltak_0;
        dcomp sumDelta1 = 0.0;
        
        
        for (ki = 0; ki<Nk[itheta1p]; ki++){ // kp along the 2 w_D wide line
            krow = ki+arrayBuffer;
            
            // Calculate the Bogoliubov variables
            epsk_0 = epsK(ki, itheta1p);
            Deltak_0 = calcDelta0k(itheta1p, ki);
            theta1p = calcAnglek(itheta1p, ki);
            Ek_0 = Ek(epsk_0, Deltak_0);
            uk_0 = uk(epsk_0, Ek_0);
            vk_0 = vk(epsk_0, Ek_0, Deltak_0);
            i = index_diag(itheta1p, krow);
            
            // Add the k-th term to the sum
            sumDelta1 += dwave(theta1p)*(uk_0*vk_0*(aDa_c[i]+bDb_c[i]-dcomp(1.0))+pow(uk_0,2.0)*conj(aDbD_c[i])-pow(vk_0,2.0)*aDbD_c[i]);
            
        }
        
        sumDelta += sumDelta1/dcomp(Nk[itheta1p]);
        
    } // end of parallel region

return W*sumDelta/dcomp(Ntheta1);
}

void calcW(){
    //    int itheta1;
    //#pragma omp parallel for schedule(dynamic, 1)
    //    for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
    //    {
    //        int ki;
    //        double Deltak_0, epsk_0, Ek_0, theta1, uk_0, vk_0, sumW;
    //        sumW = 0.0;
    
    
    //        for (ki = 0; ki<Nk[itheta1]; ki++){
    //            Deltak_0 = calcDelta0k(itheta1, ki);
    //            epsk_0 = epsK(ki, itheta1);
    //            Ek_0 = Ek(epsk_0, Deltak_0);
    //            theta1 = calcAnglek(itheta1, ki);
    //            uk_0 = uk(epsk_0,Ek_0);
    //            vk_0 = vk(epsk_0,Ek_0);
    //            sumW += dwave(theta1)*uk_0*vk_0;

    //            //sumW += pow(dwave(theta1),2.0)/Ek_0;
    //        }
    //        W[itheta1] = -Delta0/sumW*Nk[itheta1];

    //        //W[itheta1] = -2.0/sumW*Nk[itheta1];
    //    }

    int itheta1;
    //    double sumW1 = 0.0;
    //#pragma omp parallel
    //    { // beginning of parallel region
    double sumW = 0.0; // parallel variable
    //#pragma omp for schedule(dynamic, 1)
    for (itheta1 = 0; itheta1<Ntheta1; itheta1++)
    {
        int ki;
        double Deltak_0, Ek_0, epsk_0, theta1, uk_0, vk_0, sumW2;
        sumW2 = 0.0;

        for (ki = 0; ki<Nk[itheta1]; ki++){
            Deltak_0 = calcDelta0k(itheta1, ki);

            //theta1 = intToAngle(itheta1);
            //Deltak_0 = dwave(theta1)*Delta0;

            epsk_0 = epsK(ki, itheta1);
            Ek_0 = Ek(epsk_0, Deltak_0);
            theta1 = calcAnglek(itheta1, ki);
            uk_0 = uk(epsk_0,Ek_0);
            vk_0 = vk(epsk_0,Ek_0, Deltak_0);
            //            sumW2 += pow(dwave(theta1),2.0)/Ek_0;
            sumW2 += dwave(theta1)*uk_0*vk_0;
        }

//        printf("%f\n", sumW2/Nk[itheta1]);
        sumW += sumW2/Nk[itheta1];
        //W[itheta1] = -2.0/sumW*Nk[itheta1];
    }
    //#pragma omp critical
    //        {sumW1 += sumW;

    //        }
    //#pragma omp single
//    printf("%f\n", sumW);
    W = -Delta0/sumW*Ntheta1;
    //    } // end of parallel region
//    printf("W %f\n", W);

}

dcomp currentEq(double t, int idt, dcomp aDa_c_pr[], dcomp bDb_c_pr[], dcomp ab_c_pr[], dcomp aDbD_c_pr[]){
    int itheta;
    dcomp sumj = 0.0;

// #pragma omp parallel for schedule(dynamic,1) reduction(+:sumj)
    for (itheta = 0; itheta<Ntheta1; itheta++){
		int ni, ki, i;
		double theta, Deltak_0, Deltak_pr, epsk_0, epsk_pr, Ek_0, Ek_pr, uk_0, uk_pr, vk_0, vk_pr, kx, ky;
		dcomp sum2j = 0.0;     
        ky = kFy[itheta];
		for (ki = 0; ki<Nk[itheta]; ki++){
            
			Deltak_0 = calcDelta0k(itheta, ki);
			Deltak_pr = calcDelta0k(itheta, ki+1);
// 			kx = kCurrent(ki,itheta);
			epsk_0 = epsK(ki,itheta); epsk_pr = epsK_pr(ki, itheta);
			Ek_0 = Ek(epsk_0, Deltak_0); Ek_pr = Ek(epsk_pr, Deltak_pr);
			uk_0 = uk(epsk_0, Ek_0); uk_pr = uk(epsk_pr, Ek_pr);
			vk_0 = vk(epsk_0, Ek_0, Deltak_0); vk_pr = vk(epsk_pr, Ek_pr, Deltak_pr);

			i = index_pr(itheta, idt, ki+arrayBuffer);

		//                sumj += (2.0*kFy)*(aDa_c_pr[i]-bDb_c_pr[i]+kx0*q0*t0*Delta0/pow(Ek_0, 2.0)*(aDbD_c_pr[i]+ab_c_pr[i]));

			sum2j += ((uk_0*uk_pr+vk_0*vk_pr)*(aDa_c_pr[i]-bDb_c_pr[i]) +
							   (vk_0*uk_pr-uk_0*vk_pr)*(aDbD_c_pr[i]+ab_c_pr[i]));
            
//             sum2j += (Aprobe)*((uk_0*uk_pr+vk_0*vk_pr)*(aDa_c_pr[i]-bDb_c_pr[i]) +
//                     (vk_0*uk_pr-uk_0*vk_pr)*(aDbD_c_pr[i]+ab_c_pr[i]));
		}
        
        sumj += 2.0*ky*sum2j/dcomp(Nk[itheta]);
	}
    return sumj/dcomp(Ntheta1);
}

//////////////////////////////////////////////////////////////
//         Angle Functions (on the FS)
//////////////////////////////////////////////////////////////
void calcAngles(){ // must be called before any call to any FS variables
    int itheta;
    double thetap, theta, r1, r2, l1, l2, kx, ky;
       
    r1 = sqrt((Ef-hbar_wc)/t0); // inner radius
    r2 = sqrt((Ef+hbar_wc)/t0); // outter radius
    
    thetac = asin(r1/r2); // Theta-cutoff near the top
    
    // Calculate the K-points for each Theta
    for (itheta = 0; itheta<Ntheta1; itheta++){
        theta = intToAngle(itheta);
        
        if (theta < thetac || (((pi-thetac) < theta) && ((pi+thetac) > theta)) ||
                (2*pi-thetac)<theta){
            
            thetap = asin(r2*sin(theta)/r1)-theta;
            
            ky = r2*sin(theta);
            kx = r1*cos(theta+thetap);
            
            l1 = abs(kx);
            l2 = abs(r2*cos(theta));
            
            Nk[itheta] = round((l2-l1)/q0)+1;
            N[itheta] = Nk[itheta]+2*arrayBuffer;
            
            kFy[itheta] = ky;
            if (theta > pi/2.0 && theta < 3.0*pi/2.0)
                kx0[itheta] = r2*cos(theta);
            else 
                kx0[itheta] = kx;
        }
        else{
            ky = r2*sin(theta);
            kx = r2*cos(theta);
            
            l1 = abs(kx);
            
            Nk[itheta] = round(2.0*l1/q0)+1;
            N[itheta] = Nk[itheta]+2*arrayBuffer;           
           
            kFy[itheta] = ky;
            kx0[itheta] = -abs(kx);
        }
    }
    
    int idx[Ntheta1];
    size_t sizeI = sizeof(idx) / sizeof(idx[0]);  
    for (itheta = 0; itheta<Ntheta1; itheta++){
        idx[itheta] = itheta;
    }
    sort(idx,idx+sizeI,kSort);
    
    arrayIorder(Nk, idx);
    arrayIorder(N, idx);
    arrayDorder(kFy, idx);
    arrayDorder(kx0, idx);
        
    // must be called after array re-ordering!
    arrayLengths();
    
    // must be called before any calls to gapEq
    calcW();
    
    // Print to file
    FILE *Kpoints;
    Kpoints = fopen("Kpoints.txt","w");
    fprintf(Kpoints, "Theta \t");
    fprintf(Kpoints, "W \t");
    fprintf(Kpoints, "Kpoints \t");
    fprintf(Kpoints, "kx0 \t");
    fprintf(Kpoints, "kFy \n");    
    for (itheta = 0; itheta<Ntheta1; itheta++){
        
        kdotA[itheta] = 2.0*kFy[itheta]*A0eff1;
        kdotA_pr[itheta] = 2.0*kFy[itheta]*A0eff1_pr;
        
        // Define spot on Fermi Surface
        double theta = intToAngle(idx[itheta]);
        
        fprintf(Kpoints, "%f\t", theta);
        fprintf(Kpoints, "%f\t", W);
        fprintf(Kpoints, "%d\t", Nk[itheta]);
        fprintf(Kpoints, "%f\t", kx0[itheta]);
        fprintf(Kpoints, "%f\n", kFy[itheta]);
        
    }
    fclose(Kpoints);

}

// Warning! This does not indicate the angle as the indices are re-ordered!
double intToAngle(int itheta){
    double theta, thetainterval;
	
    thetainterval =  (2.0*(pi/2.0+thetac))/Ntheta1;
    theta = thetainterval*double(itheta);
    
    if (theta<pi/2.0)
        return theta;
    
    else if (theta+(pi/2.0-thetac)<3.0*pi/2.0){
        theta = theta+(pi/2.0-thetac);
        return theta;
    }
    else{
        theta = theta+2.0*(pi/2.0-thetac);
        return theta;
    }
    // atan(1.12/(sqrt((Ef/t0)-pow(1.12,2.0))));
}

////////////////////////////////////////////////////////////
//      indexing function for arrays
//          that are stored as 1D
////////////////////////////////////////////////////////////

// 3D array index
int index_p(int itheta,int ik,int idiag){
    return Ltheta[itheta]+idiag*N[itheta]+ik;
}

// 2D array index
int index_diag(int itheta, int ik){
    return L_diag[itheta]+ik;
}

// 3D array index
int index_pr(int itheta, int idt, int ik){
    return L_pr[itheta]+idt*N[itheta]+ik;
}

// Calculate the length of the array preceding each itheta section N(itheta)*Ndiag
void arrayLengths(){
    int itheta1;
    Ltheta[0] = 0;
    L_pr[0] = 0;
    L_diag[0] = 0;
    for (itheta1 = 1; itheta1<Ntheta1; itheta1++){
        Ltheta[itheta1] = Ltheta[itheta1-1] + N[itheta1-1]*Ndiag;
        L_diag[itheta1] = L_diag[itheta1-1]+N[itheta1-1];
        L_pr[itheta1] = L_pr[itheta1-1]+N[itheta1-1]*Ndt;
    }
}

dcomp calcDeltak(dcomp Delta_c, int itheta1, int ki){
    double ktheta;
    ktheta = atan((kFy[itheta1])/(kx0[itheta1]+q0*double(ki)));
    return dwave(ktheta)*Delta_c;
}

double calcDelta0k(int itheta1, int ki){
    double ktheta;
    ktheta = atan((kFy[itheta1])/(kx0[itheta1]+q0*double(ki)));
    return dwave(ktheta)*Delta0;
}

double calcAnglek(int itheta1, int ki){
    return  atan((kFy[itheta1])/(kx0[itheta1]+q0*double(ki)));
}

double calcdt(int idt){
	return dtmin+double(idt)*hdt;
}

bool kSort(int i, int j){
    return Nk[i]>Nk[j];
}

void arrayIorder(int a[],  int idx[]){
    int b[Ntheta1];
    for (int i=0; i<Ntheta1; i++){
        b[i] = a[idx[i]];
    }
    for (int i = 0; i<Ntheta1; i++){
        a[i] = b[i];
    }
}

void arrayDorder(double a[],  int idx[]){
    double b[Ntheta1];
    for (int i=0; i<Ntheta1; i++){
        b[i] = a[idx[i]];
    }
    for (int i = 0; i<Ntheta1; i++){
        a[i] = b[i];
    }
}


/////////////////////////////////////////////////
/// \brief dwave equation
/// \param theta
/// \return dwave equation
/////////////////////////////////////////////////
double dwave(double theta){
    return cos(2.0*theta+thetaFS);
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

//////////////////////////////////////////////////
void reformDiagArrays(){
    int itheta, ki, krow, i, i1;
    for (itheta =0; itheta<Ntheta1; itheta++)
        for (ki = 0; ki<Nk[itheta]; ki++){
            // check every time that we are within the buffer
            krow = ki+arrayBuffer;
            i = index_p(itheta, krow, 6);
            aDa_k[i] = aDa[i];
            bDb_k[i] = bDb[i];
            aDbD_k[i] = aDbD[i];
        }

    delete[] aDa;
    delete[] bDb;
    delete[] aDbD;
    delete[] ab;

    aDa = new dcomp[N_diag]; // Number of k points
    bDb = new dcomp[N_diag];
    aDbD = new dcomp[N_diag];
    ab = new dcomp[N_diag];
    for (itheta =0; itheta<Ntheta1; itheta++)
        for (ki = 0; ki<Nk[itheta]; ki++){
            krow = ki+arrayBuffer;
            i = index_p(itheta, krow, 6);
            i1 = index_diag(itheta, krow);
            aDa[i1] = aDa_k[i];
            bDb[i1] = bDb_k[i];
            aDbD[i1] = aDbD_k[i];
        }

    delete[] aDa_k;
    delete[] bDb_k;
    delete[] aDbD_k;
    delete[] ab_k;

    delete[] aDa_rk;
    delete[] bDb_rk;
    delete[] aDbD_rk;
    delete[] ab_rk;

    delete[] aDa_ddt;
    delete[] bDb_ddt;
    delete[] aDbD_ddt;
    delete[] ab_ddt;

    // arrays for storing the aggregate value of the current Runge-Kutta step, i: y_{n+1} = y_n + sum_i{k_i}
    aDa_rk = new dcomp[N_diag];                                                           //                          alpha-dagger alpha array
    bDb_rk = new dcomp[N_diag];                                                           //                          beta-dagger beta array
    aDbD_rk = new dcomp[N_diag];                                                       //                          alpha-dagger beta-dagger array
   
    // arrays for storing the current value of the quasiparticles for the next Runge-Kutta step, i: y_n + C*k_i where C is some constant, 1.0 or 0.5;
    aDa_k = new dcomp[N_diag];                                                           //                          alpha-dagger alpha array
    bDb_k = new dcomp[N_diag];                                                           //                          beta-dagger beta array
    aDbD_k = new dcomp[N_diag];                                                       //                          alpha-dagger beta-dagger array

    aDa_ddt = new dcomp[N_diag];
    bDb_ddt = new dcomp[N_diag];
    aDbD_ddt = new dcomp[N_diag];


}

