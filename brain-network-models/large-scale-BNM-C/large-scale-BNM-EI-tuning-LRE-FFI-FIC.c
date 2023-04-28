/*
 Brain network model with plasticity and self-tuning

 m.schirner@fu-berlin.de
 michael.schirner@bih-charite.de
 petra.ritter@bih-charite.de
 Copyright (c) 2015-2022 by Michael Schirner and Petra Ritter.
 */


#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
//#include "mpi.h"

struct Xi_p{
    float **Xi_elems;
};

struct SC_capS{
    float *cap;
}; 

struct SC_inpregS{
    int *inpreg;
};

FILE *FCout, *WFout, *WFout2, *WFout3, *WFout4, *WFout5, *WFout6, *WFout7, *WFout8, *WFoutLRE, *WFoutFFI;

#define REAL float
//#define REAL double



/* Compute Pearson's correlation coefficient */
float corr(float *x, float *y, int n){
    int i;
    float mx=0, my=0;
    
    /* Calculate the mean of the two series x[], y[] */
    for (i=0; i<n; i++) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    
    /* Calculate the correlation */
    float sxy = 0, sxsq = 0, sysq = 0, tmpx, tmpy;
    for (i=0; i<n; i++) {
        tmpx = x[i] - mx;
        tmpy = y[i] - my;
        sxy += tmpx*tmpy;
        sxsq += tmpx*tmpx;
        sysq += tmpy*tmpy;
    }
    
    return (sxy / (sqrt(sxsq)*sqrt(sysq)));
}


/* Compute root mean square error */
float rmse(float *x, float *y, int n){
    int i;
    float me=0;
    
    /* Calculate the squared difference */
    for (i=0; i<n; i++) {
        me += (x[i] - y[i]) * (x[i] - y[i]);
    }
    
    /* Calculate the mean squared difference */
    me /= n;
 
    /* Return root mean squared difference */
    return sqrt(me);
}


/* Compute mean */
float mean(float *x, int n){
    int i;
    float m=0.0f;
    
    /* Calculate mean */
    for (i=0; i<n; i++) {
        m += x[i];
    }
    m /= n;
    
    return m;
}


/* Open files for writing output */
void openFCoutfile(char *paramset){
    char outfilename[1000];memset(outfilename, 0, 1000*sizeof(char));
    char buffer[10];memset(buffer, 0, 10*sizeof(char));

    strcpy (outfilename,"output/");
    strcat (outfilename,"/BOLD_");strcat (outfilename,paramset);
    strcat (outfilename,".txt");
    FCout = fopen(outfilename, "w");
    
    memset(outfilename, 0, 1000*sizeof(char));
    strcpy (outfilename,"output/");
    strcat (outfilename,"/o0_");strcat (outfilename,paramset);
    strcat (outfilename,".txt");
    WFout = fopen(outfilename, "a");
}


float gaussrand_ret()
{
    static double V1=0.0, V2=0.0, S=0.0, U1=0.0, U2=0.0;
    S=0.0;
    do {
        U1 = (double)rand() / RAND_MAX;
        U2 = (double)rand() / RAND_MAX;
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);
    
    return (float)(V1 * sqrt(-2 * log(S) / S));
}

static inline void gaussrand(float *randnum)
{
    static double V1=0.0, V2=0.0, S=0.0, U1=0.0, U2=0.0;
    
    S=0.0;
    do {
        U1 = (double)rand() / RAND_MAX;
        U2 = (double)rand() / RAND_MAX;
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);
    
    randnum[0] = (float)(V1 * sqrt(-2 * log(S) / S));
    randnum[1] = (float)(V2 * sqrt(-2 * log(S) / S));
    
    S=0.0;
    do {
        U1 = (double)rand() / RAND_MAX;
        U2 = (double)rand() / RAND_MAX;
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);
    
    randnum[2] = (float)(V1 * sqrt(-2 * log(S) / S));
    randnum[3] = (float)(V2 * sqrt(-2 * log(S) / S));
}


int importGlobalConnectivity(char *LRE_filename, char *FFI_filename, char *SC_cap_filename, char *SC_dist_filename, char *SC_inputreg_filename, char *FC_filename, char *FCfull_filename, int nodes, float **region_activity, struct Xi_p **reg_globinp_p, float global_trans_v, int **n_conn_table, float G_J_NMDA, int **reg_idx_table, struct SC_capS **SC_cap_orig, struct SC_capS **SC_cap_LRE_frac, struct SC_capS **SC_cap_FFI_frac, struct SC_capS **SC_cap, struct SC_capS **SC_cap_FFI, float **SC_rowsums, struct SC_inpregS **SC_inpreg, float **sim_FC_vec, float **sim_FC, float **emp_FC_vec, float **emp_FC, int *num_total_connectionsp, float *mean_FCemp_abs, struct SC_capS **SC_cap_LRE_mean, struct SC_capS **SC_cap_FFI_mean, int isRestart)
{
    
    int i,j,k, tmp3, maxdelay=0, tmpint;
    float *region_activity_p;
    double tmp, tmp2, tmpFC;
    struct Xi_p *reg_globinp_pp;
    struct SC_capS      *SC_capp, *SC_capp_FFI, *SC_capp_orig, *SC_capp_LRE_frac, *SC_capp_FFI_frac, *SC_capp_LRE_mean, *SC_capp_FFI_mean;
    struct SC_inpregS   *SC_inpregp;
    int num_incoming_conn_cap=0, num_incoming_conn_dist=0, num_incoming_conn_inputreg=0, num_incoming_conn_FC=0, num_total_connections=0;
    
    // Open SC files
    FILE *file_cap, *file_dist, *file_inputreg, *file_FC, *file_FCfull, *file_LRE, *file_FFI;
    file_cap        = fopen(SC_cap_filename, "r");
    file_dist       = fopen(SC_dist_filename, "r");
    file_inputreg   = fopen(SC_inputreg_filename, "r");
    file_FC         = fopen(FC_filename, "r");
    file_FCfull     = fopen(FCfull_filename, "r");
    if (file_cap==NULL || file_dist==NULL || file_inputreg==NULL || file_FC==NULL || file_FCfull==NULL)
    {
        printf( "\nERROR: Could not open SC/FC files.\n%s\n%s\n Terminating... \n\n", LRE_filename, FFI_filename);
        exit(0);
    }
    
    if (isRestart > 0) {
        file_LRE        = fopen(LRE_filename, "r");
        file_FFI        = fopen(FFI_filename, "r");
        if (file_LRE==NULL || file_FFI==NULL) {
            printf( "\nERROR: Could not open restart files. Terminating... \n\n");
            exit(0);
        }
    }

    
    
    /*
     =========== FC ====================================================================================================================================
     */
    // Create FC vectors for computing FCsim-FCemp correlation
    int FC_subdiag_len  = (nodes*nodes - nodes) / 2;
    *sim_FC_vec         = (float *)_mm_malloc(FC_subdiag_len*sizeof(float),16);
    *emp_FC_vec         = (float *)_mm_malloc(FC_subdiag_len*sizeof(float),16);
    *sim_FC             = (float *)_mm_malloc(nodes*nodes*sizeof(float),16);
    *emp_FC             = (float *)_mm_malloc(nodes*nodes*sizeof(float),16);
    
    for (i=0; i<nodes; i++) {
        (*sim_FC_vec)[i] = 0.0f;
        (*emp_FC_vec)[i] = 0.0f;
    }
    for (i=0; i<nodes*nodes; i++) {
        (*sim_FC)[i] = 0.0f;
        (*emp_FC)[i] = 0.0f;
    }

    int fc_count = 0;
    float FC_vec_i = 0.0f;
    *mean_FCemp_abs = 0.0f;
    for (i=0; i<nodes; i++) {
        for (j=0; j<nodes; j++) {
            if(fscanf(file_FCfull,"%lf",&tmpFC) == EOF){
                printf( "\nERROR: FC full file too small. Terminating... \n\n");exit(0);
            }
            
            (*emp_FC)[i*nodes + j] = (float)tmpFC;
            
            if (i<nodes-1 && j > i) {
                (*emp_FC_vec)[fc_count] = (float)tmpFC;
                fc_count++;
                FC_vec_i += 1.0f;
                *mean_FCemp_abs        += fabsf((float)tmpFC);
            }
        }
    }
    *mean_FCemp_abs /= FC_vec_i;
    /*
     =========== /FC ====================================================================================================================================
     */
    
    
    
    
    /*
     =========== SC ====================================================================================================================================
     */
    
    // Read number of nodes in header and check whether it's consistent with other specifications
    int readSC_cap, readSC_dist, readSC_inp, readFC;
    if(fscanf(file_cap,"%d",&readSC_cap) == EOF || fscanf(file_dist,"%d",&readSC_dist) == EOF || fscanf(file_inputreg,"%d",&readSC_inp) == EOF || fscanf(file_FC,"%d",&readFC) == EOF){
        printf( "\nERROR: Unexpected end-of-file in file. File contains less input than expected. Terminating... \n\n");
        exit(0);
    }
    if (readSC_cap != nodes || readSC_dist != nodes || readSC_inp != nodes || readFC != nodes) {
        printf( "\nERROR: Inconsistent number of large-scale nodes in SC files. Terminating... \n\n");
        fclose(file_dist);fclose(file_inputreg);
        exit(0);
    }
    
    // Allocate a counter that stores number of region input connections for each region and the SCcap array
    *SC_rowsums             = (float *)_mm_malloc(nodes*sizeof(float),16);
    *n_conn_table           = (int *)_mm_malloc(nodes*sizeof(int),16);
    *reg_idx_table          = (int *)_mm_malloc(nodes*nodes*sizeof(int),16);
    int *reg_idx_tablep     = *reg_idx_table;
    *SC_cap_orig            = (struct SC_capS *)_mm_malloc(nodes*sizeof(struct SC_capS),16);
    SC_capp_orig            = *SC_cap_orig;
    *SC_cap_LRE_frac        = (struct SC_capS *)_mm_malloc(nodes*sizeof(struct SC_capS),16);
    SC_capp_LRE_frac        = *SC_cap_LRE_frac;
    *SC_cap_FFI_frac        = (struct SC_capS *)_mm_malloc(nodes*sizeof(struct SC_capS),16);
    SC_capp_FFI_frac        = *SC_cap_FFI_frac;
    *SC_cap_LRE_mean        = (struct SC_capS *)_mm_malloc(nodes*sizeof(struct SC_capS),16);
    SC_capp_LRE_mean        = *SC_cap_LRE_mean;
    *SC_cap_FFI_mean        = (struct SC_capS *)_mm_malloc(nodes*sizeof(struct SC_capS),16);
    SC_capp_FFI_mean        = *SC_cap_FFI_mean;
    *SC_cap                 = (struct SC_capS *)_mm_malloc(nodes*sizeof(struct SC_capS),16);
    SC_capp                 = *SC_cap;
    *SC_cap_FFI             = (struct SC_capS *)_mm_malloc(nodes*sizeof(struct SC_capS),16);
    SC_capp_FFI             = *SC_cap_FFI;
    *SC_inpreg              = (struct SC_inpregS *)_mm_malloc(nodes*sizeof(struct SC_inpregS),16);
    SC_inpregp              = *SC_inpreg;

    if(*n_conn_table==NULL || SC_capp_LRE_frac==NULL || SC_capp_FFI_frac==NULL || SC_capp_LRE_mean==NULL || SC_capp_FFI_mean==NULL || SC_capp_orig==NULL || SC_capp==NULL || SC_capp_FFI==NULL || SC_rowsums==NULL || SC_inpregp==NULL){
        printf("Running out of memory. Terminating.\n");fclose(file_dist);fclose(file_cap);fclose(file_inputreg);fclose(file_FC);exit(2);
    }
    
    // Read the maximal SC length in header of SCdist-file and compute maxdelay
    if(fscanf(file_dist,"%lf",&tmp) == EOF){
        printf( "ERROR: Unexpected end-of-file in file. File contains less input than expected. Terminating... \n");
        exit(0);
    }
    maxdelay = (int)((tmp/global_trans_v)+0.5); // +0.5 for rounding by casting
    if (maxdelay < 1) maxdelay = 1; // Case: no time delays
    
    // Allocate ringbuffer that contains region activity for each past time-step until maxdelay and another ringbuffer that contains pointers to the first ringbuffer
    *region_activity = (float *)_mm_malloc(maxdelay*nodes*sizeof(float),16);
    region_activity_p = *region_activity;
    *reg_globinp_p = (struct Xi_p *)_mm_malloc(maxdelay*nodes*sizeof(struct Xi_p),16);
    reg_globinp_pp = *reg_globinp_p;
    if(region_activity_p==NULL || reg_globinp_p==NULL){
        printf("Running out of memory. Terminating.\n");fclose(file_dist);exit(2);
    }
    for (j=0; j<maxdelay*nodes; j++) {
        region_activity_p[j]=0.001;
    }
    
    // Read SC files and set pointers for each input region and correspoding delay for each ringbuffer time-step
    int ring_buff_position;
    for (i=0; i<nodes; i++) {
        // Read region index of current region (first number of each row) and check whether its consistent for all files
        if(fscanf(file_cap,"%d",&num_incoming_conn_cap) == EOF || fscanf(file_dist,"%d",&num_incoming_conn_dist) == EOF || fscanf(file_inputreg,"%d",&num_incoming_conn_inputreg) == EOF || fscanf(file_FC,"%d",&num_incoming_conn_FC) == EOF){
            printf( "ERROR: Unexpected end-of-file in SC files. File(s) contain(s) less input than expected. Terminating... \n");
            exit(0);
        }
        if (num_incoming_conn_cap != i || num_incoming_conn_dist != i || num_incoming_conn_inputreg != i || num_incoming_conn_FC != i) {
            printf( "ERROR: Inconsistencies in global input files, seems like row number is incorrect in some files. i=%d cap=%d dist=%d inp=%d Terminating.\n\n", i, num_incoming_conn_cap, num_incoming_conn_dist, num_incoming_conn_inputreg);
            exit(0);
        }
        
        // Read number of incoming connections for this region (second number of each row) and check whether its consistent across input files
        if(fscanf(file_cap,"%d",&num_incoming_conn_cap) == EOF || fscanf(file_dist,"%d",&num_incoming_conn_dist) == EOF || fscanf(file_inputreg,"%d",&num_incoming_conn_inputreg) == EOF || fscanf(file_FC,"%d",&num_incoming_conn_FC) == EOF){
            printf( "ERROR: Unexpected end-of-file in file %s or %s. File contains less input than expected. Terminating... \n\n", SC_dist_filename, SC_inputreg_filename);
            exit(0);
        }
        if (num_incoming_conn_cap != num_incoming_conn_inputreg || num_incoming_conn_dist != num_incoming_conn_inputreg || num_incoming_conn_cap != num_incoming_conn_FC) {
            printf( "ERROR: Inconsistencies in SC files: Different numbers of input connections. Terminating.\n\n");
            exit(0);
        }
        
        (*n_conn_table)[i]      = num_incoming_conn_inputreg;
        
        if ((*n_conn_table)[i] > 0) {
            // SC strength and inp region numbers
            SC_capp_LRE_frac[i].cap = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_capp_FFI_frac[i].cap = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_capp_LRE_mean[i].cap = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_capp_FFI_mean[i].cap = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_capp_orig[i].cap     = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_capp[i].cap          = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_capp_FFI[i].cap      = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_inpregp[i].inpreg    = (int *)_mm_malloc(((*n_conn_table)[i])*sizeof(int),16);
            if(SC_capp_LRE_mean[i].cap==NULL || SC_capp_FFI_mean[i].cap==NULL || SC_capp_LRE_frac[i].cap==NULL || SC_capp_FFI_frac[i].cap==NULL || SC_capp_orig[i].cap==NULL || SC_capp[i].cap==NULL || SC_capp_FFI[i].cap==NULL ||  SC_inpregp[i].inpreg==NULL){
                printf("Running out of memory. Terminating.\n");exit(2);
            }
            
            // Allocate memory for input-region-pointer arrays for each time-step in ringbuffer
            for (j=0; j<maxdelay; j++){
                reg_globinp_pp[i+j*nodes].Xi_elems=(float **)_mm_malloc(((*n_conn_table)[i])*sizeof(float *),16);
                if(reg_globinp_pp[i+j*nodes].Xi_elems==NULL){
                    printf("Running out of memory. Terminating.\n");exit(2);
                }
            }
            
            float sum_caps=0.0f;
            float LRE_val=0.0f, FFI_val=0.0f;
            // Read incoming connections and set pointers
            for (j=0; j<(*n_conn_table)[i]; j++) {
                
                if(fscanf(file_cap,"%lf",&tmp) != EOF && fscanf(file_dist,"%lf",&tmp2) != EOF && fscanf(file_inputreg,"%d",&tmp3) != EOF && fscanf(file_FC,"%lf",&tmpFC) != EOF){
                    
                    
                    if (isRestart > 0) {
                        if(fscanf(file_LRE,"%f",&LRE_val) == EOF || fscanf(file_FFI,"%f",&FFI_val) == EOF) {
                            printf( "\nERROR: Unexpected end-of-file. File(s) contain(s) less input than expected. Terminating... \n\n");
                            exit(0);
                        }
                    } else {
//                        double tmp_weight = rand()/(((double)RAND_MAX + 1) / 2.0) - 1.0;
//                        LRE_val = 1.0 + tmp_weight;
//                        FFI_val = 1.0 - tmp_weight;
                      
                        LRE_val = 1.0;
                        FFI_val = 1.0;
                    }
                    
                    
                    num_total_connections++;
                    tmpint = (int)((tmp2/global_trans_v)+0.5); //+0.5 for rounding by casting
                    if (tmpint < 0 || tmpint > maxdelay){
                        printf( "\nERROR: Negative or too high (larger than maximum specified number) connection length/delay %d -> %d. Terminating... \n\n",i,tmp3);exit(0);
                    }
                    if (tmpint <= 0) tmpint = 1; // If time delay is smaller than integration step size, then set time delay to one integration step
                                       
                    SC_capp_orig[i].cap[j]     = (float)tmp * G_J_NMDA;
                    SC_capp[i].cap[j]          = (float)tmp * G_J_NMDA * LRE_val;
                    SC_capp_FFI[i].cap[j]      = (float)tmp * G_J_NMDA * FFI_val;
                    SC_capp_LRE_frac[i].cap[j] = LRE_val;
                    SC_capp_FFI_frac[i].cap[j] = FFI_val;
                    SC_capp_LRE_mean[i].cap[j] = 0.0f;
                    SC_capp_FFI_mean[i].cap[j] = 0.0f;

                    sum_caps                += SC_capp[i].cap[j];
                    SC_inpregp[i].inpreg[j]  = tmp3;
                    reg_idx_tablep[i*nodes + tmp3] = j; // This table associates region-ids (tmp3) with their positions (j) in SC_cap structs. For example a connection between region 0 and region 1 would be stored at SC_cap[0].cap[0], because SC_cap is a sparse matrix format and there is no recurrent connection between region 0 and itself. To find the index j, look at reg_idx_table[0 * n + 1]
                    
                    if (tmp3 >= 0 && tmp3 < nodes) {
                        ring_buff_position=maxdelay*nodes - tmpint*nodes + tmp3;
                        for (k=0; k<maxdelay; k++) {
                            reg_globinp_pp[i+k*nodes].Xi_elems[j]=&region_activity_p[ring_buff_position];
                            ring_buff_position += nodes;
                            if (ring_buff_position > (maxdelay*nodes-1)) ring_buff_position -= maxdelay*nodes;
                        }
                    } else {
                        printf( "\nERROR: Region index is negative or too large: %d -> %d. Terminating... \n\n",i,tmp3);exit(0);
                    }
                    
                    
                } else{
                    printf( "\nERROR: Unexpected end-of-file in file %s or %s. File contains less input than expected. Terminating... \n\n", SC_inputreg_filename, SC_dist_filename);
                    exit(0);
                }
                
            }
            if (sum_caps <= 0) {
                printf( "\nERROR: Sum of connection strenghts is negative or zero. sum-caps node %d = %f. Terminating... \n\n",i,sum_caps);exit(0);
            }
            (*SC_rowsums)[i] = sum_caps;
        }
    }
    *num_total_connectionsp = num_total_connections;
    
    fclose(file_dist);fclose(file_inputreg);fclose(file_cap);fclose(file_FC);fclose(file_FCfull);
    
    if (isRestart > 0) {
        fclose(file_LRE);fclose(file_FFI);
    }
    
    return maxdelay;
}


/*
 Plasticity rule from Zenke et al. (2015) Nat.Comm.  Eqs. 5 - 8 Suppl.
 
 % Integration of weight evolution
 dw =  A * tau_plus * tau_slow * nu_x .* nu_y.^2     ... % triplet LTP
 - A * tau_minus           * nu_x .* nu_y            ... % doublet LTD
 - beta  * (w - w_tilde)   .* chi_nu(nu_y)           ... % heterosynaptic
 + delta                   * nu_x;                       % transmitter induced
 
 w_res = w + dt .* dw;
 
 
 % burst detector Chi
 tau = 20 / 1000; % [s] membrane time constant
 chi = 1/6 * (2 + 9 * nu * tau + 6 * nu.^2 * tau^2) .* nu.^2;
 
 
 r_x:    presynaptic population firing rate
 r_y:    postsynaptic population firing rate
 
 */
//static inline void Zenke2015_SSE(const __m128 *_nu_x, const __m128 *_nu_y, __m128 *_w, const __m128 *_w_tilde, const int nodes_vec, const __m128 _dt)
//{
//    /*
//     Plasticity parameters: Zenke et al. (2015), Nat.Comm. (mean-field version)
//     */
//    const float A             = 1.0E-3f;            // [1] LTP/LTD rate
//    const float tau_plus      = 20.0f  / 1000.0f;   // [s] presynaptic trace for excitatory plasticity
//    const float tau_slow      = 100.0f / 1000.0f;   // [s] slow postsynaptic trace for excitatory plasticity
//    const float tau_minus     = 20.0f  / 1000.0f;   // [s] postsynaptic trace for excitatory plasticity
//    const float beta          = 1.25E-4f;           // [1] heterosynaptic plasticity strength parameter
//    const float delta         = 2.0E-5f;            // [1] transmitter induced plasticity strength
//    const float tau           = 20.0f / 1000.0f;    // [s] membrane time constant of burst detector chi(nu_y)
//
//    // Load plasticity parameters into m128 variable
//    const __m128 _A           = _mm_load1_ps(&A);
//    const __m128 _tau_plus    = _mm_load1_ps(&tau_plus);
//    const __m128 _tau_slow    = _mm_load1_ps(&tau_slow);
//    const __m128 _tau_minus   = _mm_load1_ps(&tau_minus);
//    const __m128 _beta        = _mm_load1_ps(&beta);
//    const __m128 _delta       = _mm_load1_ps(&delta);
//    const __m128 _tau         = _mm_load1_ps(&tau);
//
//    // Helper-variables
//    const float h_var1 = 1.0f/6.0f;
//    const float h_var2 = 2.0f;
//    const float h_var3 = 9.0f;
//    const float h_var4 = 6.0f;
//
//
//    int i;
//    __m128 _dw, _chi_nu_y;
//
//    for (i = 0; i < nodes_vec; i++) {
//        _chi_nu_y = _mm_mul_ps(_mm_mul_ps(_mm_load1_ps(&h_var1), (_mm_add_ps(_mm_add_ps(_mm_load1_ps(&h_var2),
//                                                                                        _mm_mul_ps(_mm_load1_ps(&h_var3), _mm_mul_ps(_nu_y[i], _tau))),
//                                                                             _mm_mul_ps(_mm_load1_ps(&h_var4), _mm_mul_ps(_mm_mul_ps(_nu_y[i], _nu_y[i]), _mm_mul_ps(_tau,_tau)))))),
//                               _mm_mul_ps(_nu_y[i],_nu_y[i]));
//
//        _dw =  _mm_add_ps(_mm_sub_ps(_mm_sub_ps(_mm_mul_ps(_mm_mul_ps(_A, _tau_plus), _mm_mul_ps(_mm_mul_ps(_tau_slow, _nu_x[i]), _mm_mul_ps(_nu_y[i], _nu_y[i]))),
//                                                _mm_mul_ps(_mm_mul_ps(_A, _tau_minus), _mm_mul_ps(_nu_x[i], _nu_y[i]))),
//                                     _mm_mul_ps(_mm_mul_ps(_beta, (_mm_sub_ps(_w[i], _w_tilde[i]))), _chi_nu_y)),
//                          _mm_mul_ps(_delta, _nu_x[i]));
//
//        _w[i] = _mm_add_ps(_w[i], _mm_mul_ps(_dt, _dw));
//    }
//}



/*
 Usage: tvbii <paramfile> <subject_id>
 
 e.g. 
 $ ./tvbii param_set_1 116726
 */

int main(int argc, char *argv[])
{
    /* Get current time */
    time_t start = time(NULL);
    
    
    
    /*  Open output file(s)  */
    if (argc < 6 || argc > 7) {
        printf( "\nERROR: Wrong number of arguments.\n\nUsage: tvbii <paramfile> <subid> <rand_seed> <cc_len> <eta> [<restart>]\n\nTerminating... \n\n");
        exit(0);
    }
    openFCoutfile(argv[1]);
    int isRestart = 0;
    if (argc == 7) isRestart = 1;
    
    
    int new_seed = atoi(argv[3]);
    int BOLD_cc_len = atoi(argv[4]);
    float eta = atof(argv[5]);

    
    

    /* Initialize some variables */
    int j, k;
    
    
    /* Global model and integration parameters */
    const float dt                  = 1.0;      // Integration step length dt = 1 ms
    const float model_dt            = 0.001;    // Time-step of model (sampling-rate=1000 Hz)
    const int   vectorization_grade = 4;        // How many operations can be done simultaneously. Depends on CPU Architecture and available intrinsics.
    int         time_steps          = 667*1.94*1000;    // Simulation length
    int         FIC_time_steps      = 10 * 1000;                                        // Length of FIC simulations (default: 10 s)
    int         fake_nodes               = 84;    // Number of nodes in brain network model (including fake nodes that were added to get multiples of vectorization_grade)
    int         nodes          = 84;    // Number of actual nodes in brain network model
    float       global_trans_v      = 1.0;     // Global transmission velocity (m/s)
    float       G                   = 0.5;        // Global coupling strength
    int         BOLD_TR             = 1940;     // [ms] TR of BOLD data
    int         rand_num_seed       = 1403;    // seed for random number generator
    
    
    /*  Local model: DMF-Parameters from Deco et al. JNeuro 2014 --> those are overwritten later */
    float w_plus  = 1.4;          // local excitatory recurrence
    //float I_ext   = 0;            // External stimulation
    float J_NMDA  = 0.15;         // (nA) excitatory synaptic coupling
    //float J_i     = 1.0;          // 1 for no-FIC, !=1 for Feedback Inhibition Control
    const float a_E     = 310;          // (n/C)
    const float b_E     = 125;          // (Hz)
    const float d_E     = 0.16;         // (s)
    const float a_I     = 615;          // (n/C)
    const float b_I     = 177;          // (Hz)
    const float d_I     = 0.087;        // (s)
    const float gamma   = 0.641/1000.0; // factor 1000 for expressing everything in ms
    const float tau_E   = 100;          // (ms) Time constant of NMDA (excitatory)
    const float tau_I   = 10;           // (ms) Time constant of GABA (inhibitory)
    float       sigma   = 0.00316228;   // (nA) Noise amplitude
    const float I_0     = 0.382;        // (nA) overall effective external input
    const float w_E     = 1.0;          // scaling of external input for excitatory pool
    const float w_I     = 0.7;          // scaling of external input for inhibitory pool
    const float gamma_I = 1.0/1000.0;   // for expressing inhib. pop. in ms
    float       tmpJi   = 0.0;          // Feedback inhibition J_i

    
    
    
    /* Read parameters from input file. Input file is a simple text file that contains one line with parameters and white spaces in between. */
    FILE *file;
    file=fopen(argv[1], "r");
    if (file==NULL){
        printf( "\nERROR: Could not open file %s. Terminating... \n\n", argv[1]);
        exit(0);
    }
    if(fscanf(file,"%d",&nodes) != EOF && fscanf(file,"%f",&G) != EOF && fscanf(file,"%f",&J_NMDA) != EOF && fscanf(file,"%f",&w_plus) != EOF && fscanf(file,"%f",&tmpJi) != EOF && fscanf(file,"%f",&sigma) != EOF && fscanf(file,"%d",&time_steps) != EOF && fscanf(file,"%d",&FIC_time_steps) != EOF && fscanf(file,"%d",&BOLD_TR) != EOF && fscanf(file,"%f",&global_trans_v) != EOF && fscanf(file,"%d",&rand_num_seed) != EOF){
    } else{
        printf( "\nERROR: Unexpected end-of-file in file %s. File contains less input than expected. Terminating... \n\n", argv[1]);
        exit(0);
    }
    fclose(file);
    
    
    
    
    /* Add fake regions to make region count a multiple of vectorization grade */
    if (nodes % vectorization_grade != 0){
        printf( "\nWarning: Specified number of nodes (%d) is not a multiple of vectorization degree (%d). Will add some fake nodes... \n\n", nodes, vectorization_grade);
        
        int remainder   = nodes%vectorization_grade;
        if (remainder > 0) {
            fake_nodes = nodes + (vectorization_grade - remainder);
        }
    } else{
        fake_nodes = nodes;
    }
    
    
    
    /* Initialize random number generator */
    srand((unsigned)new_seed);
    
    
    
    /* Allocate and Initialize arrays and variables */
    float *S_i_E                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *S_i_I                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *r_i_E                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *r_i_I                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
        
    float *global_input             = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *global_input_FFI         = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *J_i                      = (float *)_mm_malloc(fake_nodes * sizeof(float),16);  // (nA) inhibitory synaptic coupling
    float *meanFR                   = (float *)_mm_malloc(fake_nodes * sizeof(float),16);  // summation array for mean firing rate
    float *meanFR_INH               = (float *)_mm_malloc(fake_nodes * sizeof(float),16);  // summation array for mean firing rate
    if (S_i_E == NULL || S_i_I == NULL || r_i_E == NULL || r_i_I == NULL ||  global_input == NULL || global_input_FFI == NULL || J_i == NULL || meanFR == NULL || meanFR_INH == NULL) {
        printf( "ERROR: Running out of memory. Aborting... \n");
        _mm_free(S_i_E);_mm_free(S_i_I);_mm_free(global_input);_mm_free(J_i);_mm_free(meanFR);_mm_free(meanFR_INH);
        return 1;
    }
    
    float tmpglobinput, tmpglobinput_FFI;
    int   ring_buf_pos=0;
    float tmp_exp_E[4]          __attribute__((aligned(16)));
    float tmp_exp_I[4]          __attribute__((aligned(16)));
    float rand_number[4]        __attribute__((aligned(16)));
    
    /* Balloon-Windkessel model arrays and parameters */
    float TR          = (float)BOLD_TR / 1000;                                     // (s) TR of fMRI data
    int   BOLD_ts_len = time_steps / (TR / model_dt) + 1;                          // Length of BOLD time-series written to HDD
    float *bw_x_ex    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 1 of BW-model (exc. pop.)
    float *bw_f_ex    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 2 of BW-model (exc. pop.)
    float *bw_nu_ex   = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 3 of BW-model (exc. pop.)
    float *bw_q_ex    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 4 of BW-model (exc. pop.)
    float *bw_x_in    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 1 of BW-model (inh. pop.)
    float *bw_f_in    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 2 of BW-model (inh. pop.)
    float *bw_nu_in   = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 3 of BW-model (inh. pop.)
    float *bw_q_in    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);             // State-variable 4 of BW-model (inh. pop.)
    float *BOLD_ex = (float *)_mm_malloc(nodes * BOLD_ts_len * sizeof(float),16);
    float rho = 0.34, alpha = 0.32, tau = 0.98, y = 1.0/0.41, kappa = 1.0/0.65;
    float V_0 = 0.02, k1 = 7 * rho, k2 = 2.0, k3 = 2 * rho - 0.2, ialpha = 1.0/alpha, itau = 1.0/tau, oneminrho = (1.0 - rho);
    float f_tmp;
    int   BOLD_len  = -1;
    int   ts_bold=-1,ts, i_node_vec;
    
    
    /* Derived parameters */
    const int   nodes_vec     = fake_nodes/vectorization_grade;
    const float min_d_E       = -1.0 * d_E;
    const float min_d_I       = -1.0 * d_I;
    const float imintau_E     = -1.0 / tau_E;
    const float imintau_I     = -1.0 / tau_I;
    const float w_E__I_0      = w_E * I_0;
    const float w_I__I_0      = w_I * I_0;
    const float one           = 1.0;
    const float w_plus__J_NMDA= w_plus * J_NMDA;
    const float G_J_NMDA      = G * J_NMDA;
    float mean_mean_FR=0.0f, mean_FCemp_abs = 0.0f;
    
    /* Initialize state variables / parameters */
    for (j = 0; j < fake_nodes; j++) {
        S_i_E[j]            = 0.001;
        S_i_I[j]            = 0.001;
        global_input[j]     = 0.001;
        global_input_FFI[j] = 0.001;
        meanFR[j]           = 0.0f;
        meanFR_INH[j]       = 0.0f;
        J_i[j]              = tmpJi;
    }
    
    /* Import and setup global and local connectivity  */
    int         *n_conn_table, *reg_idx_table, num_total_connections;
    float       *region_activity, *SC_rowsums, *sim_FC_vec, *emp_FC_vec, *sim_FC, *emp_FC;
    struct Xi_p *reg_globinp_p;
    struct SC_capS      *SC_cap, *SC_cap_FFI, *SC_cap_orig, *SC_cap_LRE_frac, *SC_cap_FFI_frac, *SC_cap_LRE_mean, *SC_cap_FFI_mean;
    struct SC_inpregS   *SC_inpreg;
    
    char cap_file[100];memset(cap_file, 0, 100*sizeof(char));
    strcpy(cap_file,"input/");strcat(cap_file,argv[2]);strcat(cap_file,"_SC_strengths.txt");
    char dist_file[100];memset(dist_file, 0, 100*sizeof(char));
    strcpy(dist_file,"input/");strcat(dist_file,argv[2]);strcat(dist_file,"_SC_distances.txt");
    char reg_file[100];memset(reg_file, 0, 100*sizeof(char));
    strcpy(reg_file,"input/");strcat(reg_file,argv[2]);strcat(reg_file,"_SC_regionids.txt");
    char FC_file[100];memset(FC_file, 0, 100*sizeof(char));
    strcpy(FC_file,"input/");strcat(FC_file,argv[2]);strcat(FC_file,"_FC.txt");
    char FC_full_file[100];memset(FC_full_file, 0, 100*sizeof(char));
    strcpy(FC_full_file,"input/");strcat(FC_full_file,argv[2]);strcat(FC_full_file,"_FCfull.txt");
    
    char LRE_file[100];memset(LRE_file, 0, 100*sizeof(char));
    //snprintf(LRE_file, sizeof(LRE_file), "best_LRE_%s.txt",argv[3]);
    snprintf(LRE_file, sizeof(LRE_file), "out_recent_LRE.txt");
    char FFI_file[100];memset(FFI_file, 0, 100*sizeof(char));
    //snprintf(FFI_file, sizeof(FFI_file), "best_FFI_%s.txt",argv[3]);
    snprintf(FFI_file, sizeof(FFI_file), "out_recent_FFI.txt");

    
    int         maxdelay = importGlobalConnectivity(LRE_file, FFI_file, cap_file, dist_file, reg_file, FC_file, FC_full_file, nodes, &region_activity, &reg_globinp_p, global_trans_v, &n_conn_table, G_J_NMDA, &reg_idx_table, &SC_cap_orig, &SC_cap_LRE_frac, &SC_cap_FFI_frac, &SC_cap, &SC_cap_FFI, &SC_rowsums, &SC_inpreg, &sim_FC_vec, &sim_FC, &emp_FC_vec, &emp_FC, &num_total_connections, &mean_FCemp_abs, &SC_cap_LRE_mean, &SC_cap_FFI_mean, isRestart);
    printf("Model set up.\n");
    
    int         reg_act_size = nodes * maxdelay;

    
    /*  Initialize or cast to vector-intrinsics types for variables & parameters  */
    const __m128    _dt                 = _mm_load1_ps(&dt);
    const __m128    _w_plus_J_NMDA      = _mm_load1_ps(&w_plus__J_NMDA);
    const __m128    _a_E                = _mm_load1_ps(&a_E);
    const __m128    _b_E                = _mm_load1_ps(&b_E);
    const __m128    _min_d_E            = _mm_load1_ps(&min_d_E);
    const __m128    _a_I                = _mm_load1_ps(&a_I);
    const __m128    _b_I                = _mm_load1_ps(&b_I);
    const __m128    _min_d_I            = _mm_load1_ps(&min_d_I);
    const __m128    _gamma              = _mm_load1_ps(&gamma);
    const __m128    _gamma_I            = _mm_load1_ps(&gamma_I);
    const __m128    _imintau_E          = _mm_load1_ps(&imintau_E);
    const __m128    _imintau_I          = _mm_load1_ps(&imintau_I);
    const __m128    _w_E__I_0           = _mm_load1_ps(&w_E__I_0);
    const __m128    _w_I__I_0           = _mm_load1_ps(&w_I__I_0);
    float           tmp_sigma           = sigma*dt;// pre-compute dt*sigma for the integration of sigma*randnumber in equations (9) and (10) of Deco2014
    const __m128    _sigma              = _mm_load1_ps(&tmp_sigma);
    //const __m128    _I_0                = _mm_load1_ps(&I_0);
    const __m128    _one                = _mm_load1_ps(&one);
    const __m128    _J_NMDA             = _mm_load1_ps(&J_NMDA);
    
    __m128          *_S_i_E             = (__m128*)S_i_E;
    __m128          *_S_i_I             = (__m128*)S_i_I;
    __m128          *_r_i_E             = (__m128*)r_i_E;
    __m128          *_r_i_I             = (__m128*)r_i_I;
    __m128          *_tmp_exp_E         = (__m128*)tmp_exp_E;
    __m128          *_tmp_exp_I         = (__m128*)tmp_exp_I;
    __m128          *_rand_number       = (__m128*)rand_number;
    __m128          *_global_input      = (__m128*)global_input;
    __m128          *_global_input_FFI  = (__m128*)global_input_FFI;
    __m128          *_J_i               = (__m128*)J_i;
    __m128          *_meanFR            = (__m128*)meanFR;
    __m128          *_meanFR_INH        = (__m128*)meanFR_INH;
    __m128          _tmp_I_E, _tmp_I_I;
    __m128          _tmp_H_E, _tmp_H_I;


    
    /* Load J_i from previous run */
    if (isRestart > 0) {
        char Ji_filename[100];memset(Ji_filename, 0, 100*sizeof(char));
        //snprintf(Ji_filename, sizeof(Ji_filename), "best_Ji_%s.txt",argv[3]);
        snprintf(Ji_filename, sizeof(Ji_filename), "out_recent_Ji.txt");
        FILE *Ji_file;
        Ji_file = fopen(Ji_filename, "r");
        if (Ji_file == NULL) {
            printf( "\nERROR: Could not open Ji file. Terminating... \n\n");
            exit(0);
        }
        for (j=0; j < nodes; j++) {
            if(fscanf(Ji_file,"%f",&J_i[j]) == EOF){
                printf( "\nERROR: Unexpected end of Ji file. Terminating... \n\n");
                exit(0);
            }
        }
        fclose(Ji_file);
        printf("LRE, FFI and J_i values from previous run loaded.\n");
    }

    
    
    /* Simulation starts */
    printf("Starting simulation.\n");
    
    /*
     Reset arrays
     */
    for (j = 0; j < fake_nodes; j++) {
        S_i_E[j]            = 0.001;
        S_i_I[j]            = 0.001;
        global_input[j]     = 0.001;
        global_input_FFI[j] = 0.001;
        meanFR[j]           = 0.0;
        meanFR_INH[j]       = 0.0;
    }
    ring_buf_pos        = 0;
    for (j=0; j<maxdelay*nodes; j++) {
        region_activity[j]=0.001;
    }

    
    /*
     Reset Balloon-Windkessel model parameters and arrays
     */
    for (j = 0; j < fake_nodes; j++) {
        bw_x_ex[j] = 0.0;
        bw_f_ex[j] = 1.0;
        bw_nu_ex[j] = 1.0;
        bw_q_ex[j] = 1.0;
        bw_x_in[j] = 0.0;
        bw_f_in[j] = 1.0;
        bw_nu_in[j] = 1.0;
        bw_q_in[j] = 1.0;
    }
    
    
    /* Plasticity- and tuning-related variables */
    int next_optimization_ts = 10;
    int meanFR_i    =  0;


                                                                                         
    /* Iteration over time steps starts */
    ts_bold         =   -1;
    BOLD_len        =   -1;
    for (ts = 0; ts < time_steps; ts++) {
            /*
             Compute global coupling
             */
            // 1. Time-delayed and capacity weighted long-range input for next time-step
            for(j=0; j<nodes; j++){
                tmpglobinput     = 0;
                tmpglobinput_FFI = 0;
                for (k=0; k<n_conn_table[j]; k++) {
                    tmpglobinput     += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap[j].cap[k];
                    tmpglobinput_FFI += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap_FFI[j].cap[k];
                }
                global_input[j]     = tmpglobinput;
                global_input_FFI[j] = tmpglobinput_FFI;
            }
            
            for (i_node_vec = 0; i_node_vec < nodes_vec; i_node_vec++) {
                
                // Excitatory population firing rate
                
                //_tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_eIf_e, _ext_input[ext_inp_counter])),_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec])),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])))),_b_E);
                _tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec])),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])))),_b_E);
                *_tmp_exp_E   = _mm_mul_ps(_min_d_E, _tmp_I_E);
                tmp_exp_E[0]  = tmp_exp_E[0] != 0 ? expf(tmp_exp_E[0]) : 0.9;
                tmp_exp_E[1]  = tmp_exp_E[1] != 0 ? expf(tmp_exp_E[1]) : 0.9;
                tmp_exp_E[2]  = tmp_exp_E[2] != 0 ? expf(tmp_exp_E[2]) : 0.9;
                tmp_exp_E[3]  = tmp_exp_E[3] != 0 ? expf(tmp_exp_E[3]) : 0.9;
                _tmp_H_E  = _mm_div_ps(_tmp_I_E, _mm_sub_ps(_one, *_tmp_exp_E));
                
                _meanFR[i_node_vec] = _mm_add_ps(_meanFR[i_node_vec],_tmp_H_E);
                _r_i_E[i_node_vec] = _tmp_H_E;
                
                // Inhibitory population firing rate
                //_tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_mm_add_ps(_w_I__I_0,_mm_mul_ps(_eIf_i, _ext_input[ext_inp_counter])),_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec])), _S_i_I[i_node_vec])),_b_I);
                
                //_tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_w_I__I_0,_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec])), _S_i_I[i_node_vec])),_b_I);
                _tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_mm_add_ps(_w_I__I_0,_global_input_FFI[i_node_vec]),_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec])), _S_i_I[i_node_vec])),_b_I);
                //ext_inp_counter++;
                *_tmp_exp_I   = _mm_mul_ps(_min_d_I, _tmp_I_I);
                tmp_exp_I[0]  = tmp_exp_I[0] != 0 ? expf(tmp_exp_I[0]) : 0.9;
                tmp_exp_I[1]  = tmp_exp_I[1] != 0 ? expf(tmp_exp_I[1]) : 0.9;
                tmp_exp_I[2]  = tmp_exp_I[2] != 0 ? expf(tmp_exp_I[2]) : 0.9;
                tmp_exp_I[3]  = tmp_exp_I[3] != 0 ? expf(tmp_exp_I[3]) : 0.9;
                _tmp_H_I  = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
                
                _meanFR_INH[i_node_vec] = _mm_add_ps(_meanFR_INH[i_node_vec],_tmp_H_I);
                _r_i_I[i_node_vec] = _tmp_H_I;
                
                // Compute synaptic activity
                // CAUTION: In these equations dt * sigma is pre-computed above
                
                gaussrand(rand_number);
                _S_i_I[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_I[i_node_vec]),_mm_mul_ps(_dt,_mm_add_ps(_mm_mul_ps(_imintau_I, _S_i_I[i_node_vec]),_mm_mul_ps(_tmp_H_I,_gamma_I))));
                
                gaussrand(rand_number);
                _S_i_E[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_E[i_node_vec]),_mm_mul_ps(_dt, _mm_add_ps(_mm_mul_ps(_imintau_E, _S_i_E[i_node_vec]),_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_one, _S_i_E[i_node_vec]),_gamma),_tmp_H_E))));
            }
        
            for(j=0; j<fake_nodes; j++){
                S_i_E[j] = S_i_E[j] >= 0.0 ? S_i_E[j] : 0.0;
                S_i_E[j] = S_i_E[j] <= 1.0 ? S_i_E[j] : 1.0;
                S_i_I[j] = S_i_I[j] >= 0.0 ? S_i_I[j] : 0.0;
                S_i_I[j] = S_i_I[j] <= 1.0 ? S_i_I[j] : 1.0;
            }
        

            //ext_inp_counter -= nodes_vec;
            memcpy(&region_activity[ring_buf_pos], S_i_E, nodes*sizeof( float ));
            // 3. Shift ring-buff-pos
            ring_buf_pos = ring_buf_pos<(reg_act_size-nodes) ? (ring_buf_pos+nodes) : 0;
            meanFR_i++;
        
        
        
        //ext_inp_counter += nodes_vec;
        
        /*
         Compute BOLD for that time-step (subsampled to 1 ms)
         */
        
        for (j = 0; j < nodes; j++) {
            bw_x_ex[j]  = bw_x_ex[j]  +  model_dt * (S_i_E[j] - kappa * bw_x_ex[j] - y * (bw_f_ex[j] - 1.0));
            f_tmp       = bw_f_ex[j]  +  model_dt * bw_x_ex[j];
            bw_nu_ex[j] = bw_nu_ex[j] +  model_dt * itau * (bw_f_ex[j] - powf(bw_nu_ex[j], ialpha));
            bw_q_ex[j]  = bw_q_ex[j]  +  model_dt * itau * (bw_f_ex[j] * (1.0 - powf(oneminrho,(1.0/bw_f_ex[j]))) / rho  - powf(bw_nu_ex[j],ialpha) * bw_q_ex[j] / bw_nu_ex[j]);
            bw_f_ex[j]  = f_tmp;
            
            /*
             bw_x_in[j]  = bw_x_in[j]  +  model_dt * (S_i_I[j] - kappa * bw_x_in[j] - y * (bw_f_in[j] - 1.0));
             f_tmp       = bw_f_in[j]  +  model_dt * bw_x_in[j];
             bw_nu_in[j] = bw_nu_in[j] +  model_dt * itau * (bw_f_in[j] - powf(bw_nu_in[j], ialpha));
             bw_q_in[j]  = bw_q_in[j]  +  model_dt * itau * (bw_f_in[j] * (1.0 - powf(oneminrho,(1.0/bw_f_in[j]))) / rho  - powf(bw_nu_in[j],ialpha) * bw_q_in[j] / bw_nu_in[j]);
             bw_f_in[j]  = f_tmp;
             */
        }
        
        
        /*
         for (j = 0; j < 10; j++) {
            fprintf(WFout, "%.4f ",S_i_E[j]);
         }
        
         for (j = 0; j < 10; j++) {
            fprintf(WFout, "%.4f ",S_i_I[j]);
         }
         
         for (j = 0; j < 10; j++) {
            fprintf(WFout, "%.2f ",r_i_E[j]);
         }
        
         for (j = 0; j < 10; j++) {
            fprintf(WFout, "%.2f ",r_i_I[j]);
         }
         
        fprintf(WFout, "\n");
        fflush(WFout);
        */
        
        
        
        
        /*
         Compute BOLD time step and online paramter fitting rule
         */
        ts_bold++;
        if (ts_bold % BOLD_TR == 0) {
            //printf("%.1f %% \r", ((float)ts / (float)time_steps) * 100.0f );
            BOLD_len++;
            //printf("%d   %.6f s\n", ts_bold, (float)(time(NULL) - start));
            
            /* Compute next BOLD time step */
            for (j = 0; j < nodes; j++) {
                BOLD_ex[j*BOLD_ts_len + BOLD_len] = 100 / rho * V_0 * (k1 * (1 - bw_q_ex[j]) + k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1 - bw_nu_ex[j]));
                //BOLD_in[j*BOLD_ts_len + BOLD_len] =  100 / rho * V_0 * (k1 * (1 - bw_q_ex[j]) + k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1 - bw_nu_ex[j]));
            }
            
            
            
            
            
            /*  Compute mean firing rates */
            mean_mean_FR = 0;
            for (j = 0; j < nodes; j++){
                meanFR[j]     = meanFR[j] / meanFR_i;
                meanFR_INH[j] = meanFR_INH[j] / meanFR_i;
                mean_mean_FR += meanFR[j];
                
            }
            mean_mean_FR /= nodes;
            
           
            
            /* Optimization step (Starts with some offset to have long enough BOLD to compute FC) */
            if (BOLD_len == next_optimization_ts) {
                //int BOLD_cc_len    = 4800;
                int BOLD_cc_start  = BOLD_len - BOLD_cc_len + 1;
                int BOLD_offset    = 50;
                next_optimization_ts = BOLD_len + 1;
                
                            
                
                /*
                 #################################################
                 Inhibitory synaptic plasticity from Vogels et al. Science 2011
                 Eq 1: dw = eta(pre × post – r0 × pre)
                 #################################################
                 */
                float isp_eta = 0.001;
                float isp_r0  = 4.0;
                for (j = 0; j < nodes; j++){
                    
                    float pre  = meanFR_INH[j];
                    float post = meanFR[j];
                    
                    J_i[j] +=  isp_eta * (pre * post - isp_r0 * pre);
                    J_i[j] = J_i[j] > 0 ? J_i[j] : 0; // make sure that weight is always >=0
                    
                }
                //printf("BOLD-ts: %d   meanFR: %.2f \n", BOLD_len, mean_mean_FR);

                
                
                
                /* Perform E/I-tuning */
                if (BOLD_cc_start > BOLD_offset) {
                
                    /* Compute simulated FC */
                    float f_fc = 0.0f;
                    int i_fc = 0;
                    float tmpcc, mean_FCsim_abs = 0.0f;
                    for(j=0; j < nodes-1; j++){
                        sim_FC[j*nodes + j]     = 1;
                        for (k=j+1; k < nodes; k++) {
                            
                            tmpcc = corr(&BOLD_ex[j*BOLD_ts_len + BOLD_cc_start],&BOLD_ex[k*BOLD_ts_len + BOLD_cc_start], BOLD_cc_len);
                            
                            sim_FC[j*nodes + k]     = tmpcc;
                            sim_FC[k*nodes + j]     = tmpcc;
                            sim_FC_vec[i_fc]        = tmpcc;
                            i_fc++;
                            mean_FCsim_abs         += fabsf(tmpcc);
                            f_fc += 1.0f;
                        }
                    }
                    mean_FCsim_abs /= f_fc;
                    
                    
                    
                    /*  Compute correlation between empirical and simulated FC */
                    int FC_subdiag_len  = (nodes*nodes - nodes) / 2;
                    float fc_sim_fc_emp_cc = corr(&emp_FC_vec[0],&sim_FC_vec[0],FC_subdiag_len);
                    float fc_sim_fc_emp_mse= rmse(&emp_FC_vec[0],&sim_FC_vec[0],FC_subdiag_len);
                    
                    //if (BOLD_len % 100 == 0){
//                        printf("BOLD-ts: %d   |   r(FCsim,FCemp): %.4f   |   RMSE(FCsim,FCemp): %.4f   |   meanFR: %.2f\n", BOLD_len, fc_sim_fc_emp_cc, fc_sim_fc_emp_mse, mean_mean_FR);
                        fprintf(WFout, "%d %.5f %.5f %.2f\n", BOLD_len, fc_sim_fc_emp_cc, fc_sim_fc_emp_mse, mean_mean_FR);
                        fflush(WFout);
                        
                    //}
                    
                    
                    
                    
                    /* If last iteration produced highest FC fit: write out result */
                    //if (fc_sim_fc_emp_mse < (best_fit - 0.005)) {

                    
//                    if (fc_sim_fc_emp_mse < best_fit) {
//                    //if (BOLD_len > 3000 && BOLD_len % 201 == 0) {
//                        best_fit = fc_sim_fc_emp_mse;
//
//                        char best_fit_file[100];
//                        FILE *tmp_out;
//
//                        /* Write out LRE */
//                        memset(best_fit_file, 0, 100*sizeof(char));
//                        snprintf(best_fit_file,sizeof(best_fit_file),"best_LRE_%d.txt",best_fit_i);
//                        tmp_out = fopen(best_fit_file, "w");
//                        for (j=0; j < nodes; j++) {
//                            for (k=0; k < n_conn_table[j]; k++) {
//                                fprintf(tmp_out, "%.10f ",SC_cap_LRE_frac[j].cap[k]);
//                            }
//                            fprintf(tmp_out, "\n");
//                        }
//                        fprintf(tmp_out, "\n");
//                        fflush(tmp_out);
//                        fclose(tmp_out);
//
//                        /* Write out FFI */
//                        memset(best_fit_file, 0, 100*sizeof(char));
//                        snprintf(best_fit_file,sizeof(best_fit_file),"best_FFI_%d.txt",best_fit_i);
//                        tmp_out = fopen(best_fit_file, "w");
//                        for (j=0; j < nodes; j++) {
//                            for (k=0; k < n_conn_table[j]; k++) {
//                                fprintf(tmp_out, "%.10f ",SC_cap_FFI_frac[j].cap[k]);
//                            }
//                            fprintf(tmp_out, "\n");
//                        }
//                        fprintf(tmp_out, "\n");
//                        fflush(tmp_out);
//                        fclose(tmp_out);
//
//                        /* Write out J_i */
//                        memset(best_fit_file, 0, 100*sizeof(char));
//                        snprintf(best_fit_file,sizeof(best_fit_file),"best_Ji_%d.txt",best_fit_i);
//                        tmp_out = fopen(best_fit_file, "w");
//                        for (j=0; j < nodes; j++) {
//                            fprintf(tmp_out, "%.10f\n",J_i[j]);
//                        }
//                        fprintf(tmp_out, "\n");
//                        fflush(tmp_out);
//                        fclose(tmp_out);
//
//                        /* Write out BOLD */
//                        if (BOLD_len > 4800) {
//                            memset(best_fit_file, 0, 100*sizeof(char));
//                            snprintf(best_fit_file,sizeof(best_fit_file),"best_fMRI_%d.txt",best_fit_i);
//                            tmp_out = fopen(best_fit_file, "w");
//                            for (j = 0; j < nodes; j++) {
//                                for (k = BOLD_len-4800; k < BOLD_len; k++) {
//                                    fprintf(tmp_out, "%.7f ",BOLD_ex[j*BOLD_ts_len + k]);
//                                }
//                                fprintf(tmp_out, "\n");
//                            }
//                            fprintf(tmp_out, "\n");
//                            fflush(tmp_out);
//                            fclose(tmp_out);
//                        }
//
//                        fprintf(FCout, "%d %d %d %.5f %.5f %.2f\n", best_fit_i_out, best_fit_i, BOLD_len, fc_sim_fc_emp_cc, fc_sim_fc_emp_mse, mean_mean_FR);
//                        fflush(FCout);
//                        best_fit_i++;
//                        best_fit_i_out++;
//                        if (best_fit_i >= 100) best_fit_i = 0;
//                    }

                     //Write out regular update
                    if (BOLD_len % 25 == 0) {
                        FILE *tmp_out;

                        /* Write out LRE */
                        tmp_out = fopen("out_recent_LRE.txt", "w");
                        for (j=0; j < nodes; j++) {
                            for (k=0; k < n_conn_table[j]; k++) {
                                fprintf(tmp_out, "%.5f ",SC_cap_LRE_frac[j].cap[k]);
                            }
                            fprintf(tmp_out, "\n");
                        }
                        fprintf(tmp_out, "\n");
                        fflush(tmp_out);
                        fclose(tmp_out);

                        /* Write out FFI */
                        tmp_out = fopen("out_recent_FFI.txt", "w");
                        for (j=0; j < nodes; j++) {
                            for (k=0; k < n_conn_table[j]; k++) {
                                fprintf(tmp_out, "%.5f ",SC_cap_FFI_frac[j].cap[k]);
                            }
                            fprintf(tmp_out, "\n");
                        }
                        fprintf(tmp_out, "\n");
                        fflush(tmp_out);
                        fclose(tmp_out);

                        /* Write out J_i */
                        tmp_out = fopen("out_recent_Ji.txt", "w");
                        for (j=0; j < nodes; j++) {
                            fprintf(tmp_out, "%.5f\n",J_i[j]);
                        }
                        fprintf(tmp_out, "\n");
                        fflush(tmp_out);
                        fclose(tmp_out);


                        /* Write out BOLD */
                        if (BOLD_len > 4800) {
                            tmp_out = fopen("out_recent_fMRI.txt", "w");
                            for (j = 0; j < nodes; j++) {
                                for (k = BOLD_len-4800; k < BOLD_len; k++) {
                                    fprintf(tmp_out, "%.7f ",BOLD_ex[j*BOLD_ts_len + k]);
                                }
                                fprintf(tmp_out, "\n");
                            }
                            fprintf(tmp_out, "\n");
                            fflush(tmp_out);
                            fclose(tmp_out);
                        }


                    }
                    
                    
                    /*
                     ########################
                     Online-tuning of LRE/FFI
                     ########################
                     
                     "Plasticity" rule:
                     
                     LRE:
                     dw_ij_LRE/dt = (1 - sum_weights) + nu_ij_exc * gamma * (FC_ij_emp - FC_ij_sim) + G_tune_ij
                     
                     FFI:
                     dw_ij_FFI/dt = (1 - sum_weights) - nu_ij_inh * gamma * (FC_ij_emp - FC_ij_sim) + G_tune_ij
                     
                     sum_weights = w_LRE_ij + w_FFI_ij + w_LRE_ji + w_FFI_ji
                     
                     gamma = MSE(emp_FC_i, sim_FC_i)  // MSE of row-wise FC
                     
                     nu_ij_exc = (pre × post – r0 × pre)  // Distance of the population from its target firing rate r0 (3 Hz for EXC, 10 Hz for INH)
                     
                     */
                    
                    int pos_ij_conn = 0,presyn_node  = 0, postsyn_node = 0, pos_ji_conn  = 0;
                    for(j=0; j < nodes; j++){
                        postsyn_node = j;
                        for (k=0; k < n_conn_table[j]; k++) {
                            // Only loop through upper triangular of coupling matrices and update lower triangular elements
                            if (SC_inpreg[j].inpreg[k] > j) {
                                /* Get pre- and postsynaptic regions */
                                pos_ij_conn  = k;
                                presyn_node  = SC_inpreg[j].inpreg[pos_ij_conn];
                                pos_ji_conn  = reg_idx_table[presyn_node*nodes + postsyn_node];
                                
                                /* Get relative weights of LRE and FFI in both directions (from A to B and from B to A) */
                                float w_LRE_ij = SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn];
                                float w_FFI_ij = SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn];
                                float w_LRE_ji = SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn];
                                float w_FFI_ji = SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn];
                                
                                /* Compute sum of relative LRE and FFI weights --> to get a weight noramlization term to drive sum of weights towards 1 */
                                float sum_weights =  w_LRE_ij + w_FFI_ij + w_LRE_ji + w_FFI_ji;
                                float weight_normalization_term = 1.0f - sum_weights;
                                
                                
                                /* Difference between empirical and simulated FC for the current connection */
                                //float fc_diff_ij = 0.25 * powf((emp_FC[postsyn_node*nodes+presyn_node] - sim_FC[postsyn_node*nodes+presyn_node]),3);
                                float fc_diff_ij = (emp_FC[postsyn_node*nodes+presyn_node] - sim_FC[postsyn_node*nodes+presyn_node]);
                                float fc_diff_ji = fc_diff_ij;
                                
                                /* Compute learning rate dependent on normalized RMSE between simulated and empirical row-wise-FC of the current two nodes */
                                float mse_i = rmse(&emp_FC[postsyn_node*nodes], &sim_FC[postsyn_node*nodes], nodes) / 2.0f;
                                float mse_j = rmse(&emp_FC[presyn_node*nodes],  &sim_FC[presyn_node*nodes], nodes)  / 2.0f;
                               
                               
                                
                                /* Compute online tuning / plasticity rule */
                                weight_normalization_term = 0; // CAUTION: turns off weight normalization!!
                                
                                SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn] += dt*eta*(weight_normalization_term + mse_i * fc_diff_ij);
                                SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn] += dt*eta*(weight_normalization_term - mse_i * fc_diff_ij);
                                SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn]  += dt*eta*(weight_normalization_term + mse_j * fc_diff_ji);
                                SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn]  += dt*eta*(weight_normalization_term - mse_j * fc_diff_ji);
                                
                                SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn] = SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn]>0 ? SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn] : 0;
                                SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn] = SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn]>0 ? SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn] : 0;
                                SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn] = SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn]>0 ? SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn] : 0;
                                SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn] = SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn]>0 ? SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn] : 0;
                                

                                
                                /* Update low-pass mean-approximating LRE/FFI detector */
                                /*
                                float LRE_FFI_tau = 100;
                                if (BOLD_len > 1000) {
                                    SC_cap_LRE_mean[postsyn_node].cap[pos_ij_conn] += (-SC_cap_LRE_mean[postsyn_node].cap[pos_ij_conn] + SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn] ) / LRE_FFI_tau;
                                    SC_cap_FFI_mean[postsyn_node].cap[pos_ij_conn] += (-SC_cap_FFI_mean[postsyn_node].cap[pos_ij_conn] + SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn] ) / LRE_FFI_tau;
                                    SC_cap_LRE_mean[presyn_node].cap[pos_ji_conn]  += (-SC_cap_LRE_mean[presyn_node].cap[pos_ji_conn] + SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn] ) / LRE_FFI_tau;
                                    SC_cap_FFI_mean[presyn_node].cap[pos_ji_conn]  += (-SC_cap_FFI_mean[presyn_node].cap[pos_ji_conn] + SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn] ) / LRE_FFI_tau;
                                }
                                */
                                
                                /* Replace LRE/FFI values by their mean approximation */
                                /*
                                if (BOLD_len > 40000) {
                                    SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn] = SC_cap_LRE_mean[postsyn_node].cap[pos_ij_conn];
                                    SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn] = SC_cap_FFI_mean[postsyn_node].cap[pos_ij_conn];
                                    SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn]  = SC_cap_LRE_mean[presyn_node].cap[pos_ji_conn];
                                    SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn]  = SC_cap_FFI_mean[presyn_node].cap[pos_ji_conn];
                                }
                                */
                                
                                /* Update connectivity */
                                SC_cap[postsyn_node].cap[pos_ij_conn]     =
                                    SC_cap_LRE_frac[postsyn_node].cap[pos_ij_conn] * SC_cap_orig[postsyn_node].cap[pos_ij_conn];
                                
                                SC_cap_FFI[postsyn_node].cap[pos_ij_conn] =
                                    SC_cap_FFI_frac[postsyn_node].cap[pos_ij_conn] * SC_cap_orig[postsyn_node].cap[pos_ij_conn];
                                
                                SC_cap[presyn_node].cap[pos_ji_conn]      =
                                    SC_cap_LRE_frac[presyn_node].cap[pos_ji_conn]  * SC_cap_orig[presyn_node].cap[pos_ji_conn];
                                
                                SC_cap_FFI[presyn_node].cap[pos_ji_conn]  =
                                    SC_cap_FFI_frac[presyn_node].cap[pos_ji_conn]  * SC_cap_orig[presyn_node].cap[pos_ji_conn];

                                
                            } // if j>i
                        } // for: nodes j
                    } // for: nodes i
                } // if (BOLD_cc_start > BOLD_offset)
            } // if (next_optimization_ts == -1 || BOLD_len == next_optimization_ts)
            /* Reset variables */
            for (j = 0; j < fake_nodes; j++) {
                meanFR[j]           = 0.0f;
                meanFR_INH[j]       = 0.0f;
            }
            meanFR_i = 0;
        } // if (ts_bold % BOLD_TR == 0)
    } // Simulation loop
        

    
    /*
     Compute mean firing rate
     */
    /*
    mean_mean_FR = 0;
    for (j = 0; j < nodes; j++){
        meanFR[j] = meanFR[j]*meanFRfact;
        mean_mean_FR += meanFR[j];
    }
    mean_mean_FR /= nodes;
    */
    
    
    /*
     Print fMRI time series
     */
    
    //fprintf(FCout, "%.4f %.4f %.4f %.4f %.4f %.4f %.2f\n", G, J_NMDA, w_plus, tmpJi, sigma, global_trans_v, mean_mean_FR);
    /*
    for (i=0; i<nodes; i++) {
        for (j=0; j<BOLD_len; j++) {
            fprintf(FCout, "%.7f ",BOLD_ex[i*BOLD_ts_len+j]);
        }
        fprintf(FCout, "\n");
    }
    fprintf(FCout, "\n");
    fflush(FCout);
     */
    
    _mm_free(n_conn_table);
    _mm_free(region_activity);
    _mm_free(reg_globinp_p);
    _mm_free(SC_cap);
    
    fclose(FCout);
    fclose(WFout);
    printf("E/I-tuning finished. Execution took %.2f s\n", (float)(time(NULL) - start));
    
    return 0;
}
