/*
 Brain network model with plasticity, E/I-tuning
 and decision-making according to
 
 Murray, J. D., Jaramillo, J., & Wang, X. J. (2017). Working memory and decision-making in a frontoparietal circuit model. Journal of Neuroscience, 37(50), 12167-12186.

 m.schirner@fu-berlin.de
 michael.schirner@bih-charite.de
 petra.ritter@bih-charite.de
 Copyright (c) 2015-2018 by Michael Schirner and Petra Ritter.
 */


#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
//#include "mpi.h"

struct ts{
    float *ts;
};

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

float std(float *x, int n)
{
    // Compute mean (average of elements)
    int i;
    float m=0.0f;
    
    /* Calculate mean */
    for (i=0; i<n; i++) {
        m += x[i];
    }
    m /= n;
    
    // Compute sum squared
    // differences with mean.
    float sqDiff = 0.0;
    for (i = 0; i < n; i++) {
        sqDiff += (x[i] - m) * (x[i] - m);
    }
    
    return sqrtf(sqDiff / n);
}

/* Open files for writing output */
void openFCoutfile(char *paramset){
    char outfilename[1000];memset(outfilename, 0, 1000*sizeof(char));
    char buffer[10];memset(buffer, 0, 10*sizeof(char));

    strcpy (outfilename,"output/");
    strcat (outfilename,"/BOLD_");strcat (outfilename,paramset);
    strcat (outfilename,".txt");
    //FCout = fopen(outfilename, "w");
    
    memset(outfilename, 0, 1000*sizeof(char));
    strcpy (outfilename,"output/");
    strcat (outfilename,"/o0_");strcat (outfilename,paramset);
    strcat (outfilename,".txt");
    //WFout = fopen(outfilename, "a");
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
                    
                   
                    /*
                     Noise autocorr
                     
                     w=randn(10000,1);
                     
                     x(1) = w(1);
                     
                     r=0.8;
                     
                     for ii = 1:length(w)-1
                     x(ii+1) = r * x(ii) + (1 - r^2)^0.5 * w(ii+1);    
                     end
                     */
                    
                    
  
                    
                    /*
                    FFE                        = 1.0f + (float)tmpFC;
                    FFI                        = 1.0f - (float)tmpFC;
                
                    SC_capp[i].cap[j]          = FFE * (float)tmp * G_J_NMDA;
                    SC_capp_FFI[i].cap[j]      = FFI * (float)tmp * G_J_NMDA;
                    SC_capp_LRE_frac[i].cap[j] = FFE;
                    SC_capp_FFI_frac[i].cap[j] = FFI;
                    */
                    
                    
                    SC_capp_orig[i].cap[j]     = (float)tmp * G_J_NMDA;
                    SC_capp[i].cap[j]          = (float)tmp * G_J_NMDA * LRE_val;
                    SC_capp_FFI[i].cap[j]      = (float)tmp * G_J_NMDA * FFI_val;
                    SC_capp_LRE_frac[i].cap[j] = LRE_val;
                    SC_capp_FFI_frac[i].cap[j] = FFI_val;
                    SC_capp_LRE_mean[i].cap[j] = 0.0f;
                    SC_capp_FFI_mean[i].cap[j] = 0.0f;
                    /*
                    SC_capp_LRE_frac[i].cap[j] = (0.5 / 2);
                    SC_capp_FFI_frac[i].cap[j] = (0.5 / 2);
                    SC_capp_LRE_mean[i].cap[j] = 0.0f;
                    SC_capp_FFI_mean[i].cap[j] = 0.0f;
                    */

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

static inline float MW_r(const float I) {
    const float a       = 270.0;     // (Hz/nA)
    const float b       = 108.0;     // (Hz)
    const float c       = 0.154;   // (s)
    
    return (a * I - b) / (1.0 - expf(-c * (a * I - b)));
}

static inline float fI_A1(const float S_A1, const float S_A2, const float S_B1, const float S_B2, const float I_net_bg, const float I_appA) {
    const float J_same_M1 = 0.3169;
    const float J_diff_M1 = -0.0330;
    const float J_same_M2_to_M1 = 0.02;
    const float J_diff_M2_to_M1 = -0.02;
    const float I_0 = 0.334;
    
    return (J_same_M1 * S_A1 + J_diff_M1 * S_B1 + J_same_M2_to_M1 * S_A2 + J_diff_M2_to_M1 * S_B2 + I_0 + I_net_bg + I_appA);
}

static inline float fI_B1(const float S_A1, const float S_A2, const float S_B1, const float S_B2, const float I_net_bg, const float I_appB) {
    const float J_same_M1 = 0.3169;
    const float J_diff_M1 = -0.0330;
    const float J_same_M2_to_M1 = 0.02;
    const float J_diff_M2_to_M1 = -0.02;
    const float I_0 = 0.334;
    return (J_diff_M1 * S_A1 + J_same_M1 * S_B1 + J_diff_M2_to_M1 * S_A2 + J_same_M2_to_M1 * S_B2 + I_0 + I_net_bg + I_appB);
}

static inline float fI_A2(const float S_A1, const float S_A2, const float S_B1, const float S_B2, const float I_net_bg) {
    const float J_same_M2 = 0.351;
    const float J_diff_M2 = -0.0671;
    const float J_same_M1_to_M2 = 0.075;
    const float J_diff_M1_to_M2 = -0.075;
    const float I_0 = 0.334;
    return (J_same_M1_to_M2 * S_A1 + J_diff_M1_to_M2 * S_B1 + J_same_M2 * S_A2 + J_diff_M2 * S_B2 + I_0 + I_net_bg);
}

static inline float fI_B2(const float S_A1, const float S_A2, const float S_B1, const float S_B2, const float I_net_bg) {
    const float J_same_M2 = 0.351;
    const float J_diff_M2 = -0.0671;
    const float J_same_M1_to_M2 = 0.075;
    const float J_diff_M1_to_M2 = -0.075;
    const float I_0 = 0.334;
    return (J_diff_M1_to_M2 * S_A1 + J_same_M1_to_M2 * S_B1 + J_diff_M2 * S_A2 + J_same_M2 * S_B2 + I_0 + I_net_bg);
}

// Eq. 1 MurrayWang 2017 JNeuro
static inline float S_AB(float S, const float fr) {
    const float dt = 1.0;
    const float tau = 60.0;     // (ms) NMDA time constant
    const float gamma = 0.641;  // rate of saturation of S
    S = S + dt * ( -S / tau + gamma * (1.0  -  S) * fr / 1000.0 );
    S = S > 0.0 ? S : 0.0;
    S = S < 1.0 ? S : 1.0;
    return S;
}

// Noise process from MurrayWang 2017 JNeuro
static inline float MW_noise(float I_noise) {
    const float dt = 1.0;
    const float tau_AMPA = 2.0;     // (ms) NAMPA noise  time constant
    const float sigma_noise = 0.009; // (nA) noise strength
    return I_noise + dt * (  (-I_noise + gaussrand_ret() * sqrtf(tau_AMPA  *  powf(sigma_noise,2))) / tau_AMPA );
}

// Rate detector
//static inline float rate(const float mov_avg, const float curr_fr) {
//    const float alpha = 0.00025;
//    return alpha * curr_fr + (1.0 - alpha) * mov_avg;
//}

// Detect decision time
static inline int dec_time(float *x, int n) {
    int i;
    for (i=0; i<n; i++) if (x[i] > 30) return i;
    return -1;
}


/*
 Usage: tvbii <paramfile> <subject_id>
 
 e.g. 
 $ ./tvbii param_set_1 116726
 */

int main(int argc, char *argv[])
{
    /* Get current time */
    time_t start = time(NULL);
    
    /*  Parse input arguments  */
    if (argc < 6 || argc > 7) {
        printf( "\nERROR: Wrong number of arguments.\n\nUsage: tvbii <paramfile> <subid> <start_seed> <end_seed> <contrast>\n\nTerminating... \n\n");
        exit(0);
    }
    //openFCoutfile(argv[1]);
    int isRestart = 1;
    int start_seed = atoi(argv[3]);
    int end_seed = atoi(argv[4]);
    float inp_contrast = (float)atof(argv[5]);
    int j, k;
    
    /* Global model and integration parameters */
    const float dt                  = 1.0;      // Integration step length dt = 1 ms
    const int   vectorization_grade = 4;        // How many operations can be done simultaneously. Depends on CPU Architecture and available intrinsics.
    int         time_steps          = 30000;    // Simulation length
    int         burn_in_ts          = 25000;        // Length of FIC simulations (default: 10 s)
    int         fake_nodes          = 84;    // Number of nodes in brain network model (including fake nodes that were added to get multiples of vectorization_grade)
    int         nodes               = 84;    // Number of actual nodes in brain network model
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
    //float       eIf_e   = 1.0;          // weighting factor for external input (exc. pop.)
    //float       eIf_i   = 1.0;          // weighting factor for external input (inh. pop.)
    float       tmpJi   = 0.0;          // Feedback inhibition J_i

    
    
    
    /* Read parameters from input file. Input file is a simple text file that contains one line with parameters and white spaces in between. */
    FILE *file;
    file=fopen(argv[1], "r");
    if (file==NULL){
        printf( "\nERROR: Could not open file %s. Terminating... \n\n", argv[1]);
        exit(0);
    }
    if(fscanf(file,"%d",&nodes) != EOF && fscanf(file,"%f",&G) != EOF && fscanf(file,"%f",&J_NMDA) != EOF && fscanf(file,"%f",&w_plus) != EOF && fscanf(file,"%f",&tmpJi) != EOF && fscanf(file,"%f",&sigma) != EOF && fscanf(file,"%d",&time_steps) != EOF && fscanf(file,"%d",&burn_in_ts) != EOF && fscanf(file,"%d",&BOLD_TR) != EOF && fscanf(file,"%f",&global_trans_v) != EOF && fscanf(file,"%d",&rand_num_seed) != EOF){
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
    
    
    
    /* Allocate and Initialize arrays and variables */
    float *S_i_E                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *S_i_I                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *r_i_E                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *r_i_I                    = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *amp_PFC                  = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *amp_PFCs                 = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *amp_PPC                  = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *amp_PPCs                 = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    
    int pfcppc_combinations = 9 * 10;
    float *S_A1                     = (float *)_mm_malloc(pfcppc_combinations * sizeof(float),16);
    float *S_B1                     = (float *)_mm_malloc(pfcppc_combinations * sizeof(float),16);
    float *S_A2                     = (float *)_mm_malloc(pfcppc_combinations * sizeof(float),16);
    float *S_B2                     = (float *)_mm_malloc(pfcppc_combinations * sizeof(float),16);
    struct ts *r_A1                 = (struct ts *)_mm_malloc(pfcppc_combinations * sizeof(struct ts),16);
    struct ts *r_B1                 = (struct ts *)_mm_malloc(pfcppc_combinations * sizeof(struct ts),16);
    struct ts *r_A2                 = (struct ts *)_mm_malloc(pfcppc_combinations * sizeof(struct ts),16);
    struct ts *r_B2                 = (struct ts *)_mm_malloc(pfcppc_combinations * sizeof(struct ts),16);
    for (j = 0; j < pfcppc_combinations; j++) {
        S_A1[j]             = 0.0;
        S_B1[j]             = 0.0;
        S_A2[j]             = 0.0;
        S_B2[j]             = 0.0;
        r_A1[j].ts = (float *)_mm_malloc((time_steps - burn_in_ts)  * sizeof(float),16);
        r_B1[j].ts = (float *)_mm_malloc((time_steps - burn_in_ts)  * sizeof(float),16);
        r_A2[j].ts = (float *)_mm_malloc((time_steps - burn_in_ts)  * sizeof(float),16);
        r_B2[j].ts = (float *)_mm_malloc((time_steps - burn_in_ts)  * sizeof(float),16);
        for (k = 0; k < (time_steps - burn_in_ts); k++) {
            r_A1[j].ts[k] = 0.0;
            r_B1[j].ts[k] = 0.0;
            r_A2[j].ts[k] = 0.0;
            r_B2[j].ts[k] = 0.0;
        }
    }
    
    struct ts *tmp_out = (struct ts *)_mm_malloc(fake_nodes * sizeof(struct ts),16);
    for (j = 0; j < fake_nodes; j++) {
        tmp_out[j].ts = (float *)_mm_malloc((time_steps - burn_in_ts) * sizeof(float),16);
        for (k = 0; k < (time_steps - burn_in_ts); k++) {
            tmp_out[j].ts[k] = 0.0;
        }
    }

    float *I_E                      = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *I_noiseA                 = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *I_noiseB                 = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *I_PPC_A                  = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *I_PPC_B                  = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *I_PFC_A                  = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *I_PFC_B                  = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    
    float *global_input             = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *global_input_FFI         = (float *)_mm_malloc(fake_nodes * sizeof(float),16);
    float *J_i                      = (float *)_mm_malloc(fake_nodes * sizeof(float),16);  // (nA) inhibitory synaptic coupling
    
    if (S_i_E == NULL || S_i_I == NULL || r_i_E == NULL || r_i_I == NULL ||  global_input == NULL || global_input_FFI == NULL || J_i == NULL) {
        printf( "ERROR: Running out of memory. Aborting... \n");
        _mm_free(S_i_E);_mm_free(S_i_I);_mm_free(global_input);_mm_free(J_i);
        return 1;
    }
    
    float tmpglobinput, tmpglobinput_FFI;
    int   ring_buf_pos=0;
    float tmp_exp_E[4]          __attribute__((aligned(16)));
    float tmp_exp_I[4]          __attribute__((aligned(16)));
    float rand_number[4]        __attribute__((aligned(16)));
    int   ts, i_node_vec;
    
    
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
    float mean_FCemp_abs = 0.0f;
    
    

    
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
    __m128          *_I_E               = (__m128*)I_E;
    
    __m128          *_tmp_exp_E         = (__m128*)tmp_exp_E;
    __m128          *_tmp_exp_I         = (__m128*)tmp_exp_I;
    __m128          *_rand_number       = (__m128*)rand_number;
    __m128          *_global_input      = (__m128*)global_input;
    __m128          *_global_input_FFI  = (__m128*)global_input_FFI;
    __m128          *_J_i               = (__m128*)J_i;
    __m128          _tmp_I_E, _tmp_I_I;
    __m128          _tmp_H_E, _tmp_H_I;


    
    /* Load J_i from previous run */
    for (j = 0; j < fake_nodes; j++) {
        J_i[j]              = tmpJi;
    }
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
    
    
    /* Indices of PFC and PPC regions in the SC (-1 for 0-based indexing) */
    int PPC_regs[10] = { (117-1), (95-1), (144-1), (145-1), (46-1), (29-1), (45-1), (15-1), (149-1), (151-1) };
    int PFC_regs[9] = { (85-1), (86-1), (83-1), (73-1), (97-1), (98-1), (67-1), (26-1), (63-1) };


    /* open output file */
    char desc_file[100];memset(desc_file, 0, 100*sizeof(char));
    snprintf(desc_file,sizeof(desc_file),"DMfinaloutput_seedfrom%dto%d_contrast%f.txt", start_seed, end_seed, inp_contrast);
    FILE *tmp_corrdesc = fopen(desc_file, "w");

    
    /* Simulation starts */
    printf("Starting simulation.\n");
    
    /* Iterate over rng seeds */
    while (start_seed <= end_seed) {
        int new_seed = start_seed;
        start_seed++;
        
        /*
         Reset arrays
         */
        for (j = 0; j < fake_nodes; j++) {
            S_i_E[j]            = 0.001;
            S_i_I[j]            = 0.001;
            global_input[j]     = 0.001;
            global_input_FFI[j] = 0.001;
            I_noiseA[j]           = 0.0;
            I_noiseB[j]           = 0.0;
            I_PPC_A[j]            = 0.0;
            I_PPC_B[j]            = 0.0;
            I_PFC_A[j]            = 0.0;
            I_PFC_B[j]            = 0.0;
            amp_PFC[j]            = 0.0;
            amp_PFCs[j]            = 0.0;
            amp_PPC[j]            = 0.0;
            amp_PPCs[j]            = 0.0;
        }
        ring_buf_pos        = 0;
        for (j=0; j<maxdelay*nodes; j++) {
            region_activity[j]=0.001;
        }
        for (j = 0; j < pfcppc_combinations; j++) {
            S_A1[j]             = 0.0;
            S_B1[j]             = 0.0;
            S_A2[j]             = 0.0;
            S_B2[j]             = 0.0;
            for (k = 0; k < (time_steps - burn_in_ts); k++) {
                r_A1[j].ts[k] = 0.0;
                r_B1[j].ts[k] = 0.0;
                r_A2[j].ts[k] = 0.0;
                r_B2[j].ts[k] = 0.0;
            }
        }
    
        /* Initialize random number generator */
        srand((unsigned)new_seed);
    
        /* Iteration over time steps starts */
        int s_ts = 0;
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
                // Excitatory population
                _tmp_I_E = _mm_add_ps(_mm_mul_ps(_w_plus_J_NMDA,_S_i_E[i_node_vec]),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])));
                _I_E[i_node_vec] = _tmp_I_E;
                _tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_w_E__I_0,_tmp_I_E)),_b_E);
                //_tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec])),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])))),_b_E);
                
                *_tmp_exp_E   = _mm_mul_ps(_min_d_E, _tmp_I_E);
                tmp_exp_E[0]  = tmp_exp_E[0] != 0 ? expf(tmp_exp_E[0]) : 0.9;
                tmp_exp_E[1]  = tmp_exp_E[1] != 0 ? expf(tmp_exp_E[1]) : 0.9;
                tmp_exp_E[2]  = tmp_exp_E[2] != 0 ? expf(tmp_exp_E[2]) : 0.9;
                tmp_exp_E[3]  = tmp_exp_E[3] != 0 ? expf(tmp_exp_E[3]) : 0.9;
                _tmp_H_E  = _mm_div_ps(_tmp_I_E, _mm_sub_ps(_one, *_tmp_exp_E));
                _r_i_E[i_node_vec] = _tmp_H_E;
                
                // Inhibitory population
                _tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_mm_add_ps(_w_I__I_0,_global_input_FFI[i_node_vec]),_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec])), _S_i_I[i_node_vec])),_b_I);
                *_tmp_exp_I   = _mm_mul_ps(_min_d_I, _tmp_I_I);
                tmp_exp_I[0]  = tmp_exp_I[0] != 0 ? expf(tmp_exp_I[0]) : 0.9;
                tmp_exp_I[1]  = tmp_exp_I[1] != 0 ? expf(tmp_exp_I[1]) : 0.9;
                tmp_exp_I[2]  = tmp_exp_I[2] != 0 ? expf(tmp_exp_I[2]) : 0.9;
                tmp_exp_I[3]  = tmp_exp_I[3] != 0 ? expf(tmp_exp_I[3]) : 0.9;
                _tmp_H_I  = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
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

            

            
            // Integration of all Murraywang PPC/PFC combinations
            if (ts >= burn_in_ts) {
                 /*
                 Integrate small-scale model
                 */
                int i;
                // normalization parameters for each region
                float a_old[379] = {-0.0427745533333333,-0.0154646866666667,-0.0222730875000000,-0.101815385833333,-0.0573155483333333,-0.0652734975000000,-0.0138233633333333,-0.0418045516666667,-0.0498116608333333,-0.0188670058333333,-0.0130252266666667,-0.0100139466666667,-0.0315693283333333,-0.00479473500000000,-0.0161076825000000,-0.0179460166666667,-0.0217881208333333,-0.0188632116666667,-0.0198762991666667,-0.0186556950000000,-0.0122057841666667,-0.0136277883333333,-0.0122954866666667,-0.00857557833333333,-0.0203739241666667,-0.00343477083333333,-0.00997493833333333,-0.0174031900000000,-0.00853531166666667,-0.00985897083333333,-0.0131887650000000,-0.00381692750000000,-0.00217864333333333,-0.00237580333333333,-0.00575632416666667,-0.00769253083333333,-0.0115928525000000,-0.0130536241666667,-0.00921923250000000,-0.00761533333333333,-0.00743188500000000,-0.0111487575000000,-0.0133290033333333,-0.0123385216666667,-0.0167258766666667,-0.0142643450000000,-0.0153024000000000,-0.0215005625000000,-0.0140525150000000,-0.0238481941666667,-0.0361785425000000,-0.0410206600000000,-0.0278907916666667,-0.00730168250000000,-0.00972776750000000,-0.0127562358333333,-0.00572754833333333,-0.00249243166666667,-0.00551618416666667,-0.0105462450000000,-0.00190584500000000,-0.00620646916666667,-0.0105700058333333,-0.00401122166666667,-0.00414160333333333,-0.00275670166666667,-0.0101718566666667,-0.00939562166666667,-0.0194864833333333,-0.00638748000000000,-0.00519854166666667,-0.00779003250000000,-0.0173877425000000,-0.00904933833333333,-0.00787596333333333,-0.00698515833333333,-0.00823386083333333,-0.0223545991666667,-0.0101265958333333,-0.00785784250000000,-0.0157032208333333,-0.0170276733333333,-0.0188340475000000,-0.0234342958333333,-0.00871865666666667,-0.0186330433333333,-0.00572679250000000,-0.00363618750000000,-0.00393742833333333,-0.00189175833333333,-0.00494044833333333,-0.00329119500000000,-0.000295151666666667,-0.00247106250000000,-0.0133470508333333,-0.0172676475000000,-0.00676912916666667,-0.00264526416666667,-0.0102378466666667,-0.0160306541666667,-0.00705441083333333,-0.00539997750000000,-0.00315805916666667,-0.0159316116666667,-0.0131662325000000,-0.00669796166666667,-0.00358091250000000,-0.00964053000000000,-0.00742234416666667,-0.00215743333333333,-0.00518159333333333,-0.000602047500000000,-0.00527543333333333,-0.00425522000000000,-0.00394432500000000,-0.0148014500000000,-0.0274749600000000,-0.000869718333333333,-0.00460579250000000,-0.00597069000000000,-0.00865953166666667,-0.00250547333333333,-0.00245766583333333,-0.0136469766666667,-0.0132609716666667,-0.00377844166666667,-0.00559648083333333,-0.00654731666666667,-0.00978386250000000,-0.0105432166666667,-0.00627056916666667,-0.00192784333333333,-0.0169275383333333,-0.00417408166666667,-0.00450355916666667,-0.0119137608333333,-0.0220079191666667,-0.0228346333333333,-0.0265514050000000,-0.0206018383333333,-0.0161063350000000,-0.0215018475000000,-0.0300319541666667,-0.0180267183333333,-0.0188088758333333,-0.0230095608333333,-0.0172336691666667,-0.0386225091666667,-0.0279873891666667,-0.0248552475000000,-0.0235898366666667,-0.0164231783333333,-0.0101996791666667,-0.00922175333333333,-0.00266604333333333,-0.0151134541666667,-0.0162143841666667,-0.0216175225000000,-0.0212425650000000,-0.00859897000000000,-0.00279909166666667,-0.00501428916666667,-0.0107900225000000,0.00138332833333333,-0.00131901666666667,-0.000408385000000000,-0.00637661750000000,-0.00500678833333333,-0.00848466916666667,-0.00843478916666667,-0.00947315583333333,-0.00278312083333333,-0.00471278916666667,-0.0108830108333333,-0.0216855358333333,-0.00337015083333333,-0.00286502583333333,-0.00292951500000000,-0.00789028166666667,-0.00247558666666667,-0.0471953375000000,-0.0136476841666667,-0.0189970383333333,-0.117515082500000,-0.0606997683333333,-0.0641473725000000,-0.0117388525000000,-0.0596659641666667,-0.0629938341666667,-0.0175988616666667,-0.0107891950000000,-0.0106252325000000,-0.0292950325000000,-0.00358198583333333,-0.0135941483333333,-0.0160783025000000,-0.0237543016666667,-0.0165828491666667,-0.0167470091666667,-0.0142042500000000,-0.0156249391666667,-0.0117999408333333,-0.0128500400000000,-0.00910462833333333,-0.0119669283333333,-0.00464957333333333,-0.0100513866666667,-0.0140663100000000,-0.00804425833333333,-0.0114671858333333,-0.0104633758333333,-0.00275100333333333,-0.00330041916666667,-0.00494538416666667,-0.00726741916666667,-0.00736409416666667,-0.0118554900000000,-0.0157250733333333,-0.00919416166666667,-0.00852293583333334,-0.00860068333333333,-0.0133555008333333,-0.0133478025000000,-0.0102398825000000,-0.0167802958333333,-0.0159541558333333,-0.0155807291666667,-0.0159084125000000,-0.0148165550000000,-0.0213162358333333,-0.0404774250000000,-0.0434680050000000,-0.0353999108333333,-0.00733659916666667,-0.00918128666666667,-0.0103539775000000,-0.00425442666666667,-0.00142742916666667,-0.00547279416666667,-0.00991740333333333,-0.00175770583333333,-0.00660570416666667,-0.0101501316666667,-0.00332319250000000,-0.00580877083333333,-0.00367491833333333,-0.0123649975000000,-0.00920374666666667,-0.0183572000000000,-0.00374728750000000,-0.00504521166666667,-0.0133265625000000,-0.0209734958333333,-0.00993385000000000,-0.0127514933333333,-0.00735716333333333,-0.00982640083333333,-0.0213026566666667,-0.00973197000000000,-0.00929763833333333,-0.0117341350000000,-0.0180645858333333,-0.0165221908333333,-0.0220542566666667,-0.0122312333333333,-0.0220756450000000,-0.00853024833333333,-0.00305944166666667,-0.00463232583333333,-0.000988369166666667,-0.00787721666666667,-0.00382976833333333,-0.00155338833333333,-0.00230995916666667,-0.0137589891666667,-0.0185189008333333,-0.00760710750000000,-0.00331944583333333,-0.0105227066666667,-0.0174265300000000,-0.00911874166666667,-0.00827966000000000,-0.00416830166666667,-0.0119088216666667,-0.0128263200000000,-0.00746481083333333,-0.00441454916666667,-0.0114724091666667,-0.00695119583333333,-0.00409846166666667,-0.00472392833333333,-0.00222184583333333,-0.00531598083333333,-0.00520256500000000,-0.00555425833333333,-0.0167717466666667,-0.0233677383333333,-0.000508479166666667,-0.00225157583333333,-0.00476944500000000,-0.00796273833333333,-0.00244214666666667,-0.00282423166666667,-0.0172125758333333,-0.0123561758333333,-0.00320655166666667,-0.00462587416666667,-0.00555572833333333,-0.0128227100000000,-0.0138571500000000,-0.00622488750000000,-0.00182782583333333,-0.0162378858333333,-0.00327781750000000,-0.00281184500000000,-0.0104239008333333,-0.0228345708333333,-0.0269618041666667,-0.0136623116666667,-0.0174896991666667,-0.0121271050000000,-0.0177400900000000,-0.0269664916666667,-0.0160804008333333,-0.0150890700000000,-0.0192322558333333,-0.0180809975000000,-0.0324281141666667,-0.0320185016666667,-0.0262846883333333,-0.0246349950000000,-0.0168963025000000,-0.00845096583333333,-0.00789340333333333,-0.00156870583333333,-0.0144777200000000,-0.0143173166666667,-0.0173580800000000,-0.0163798075000000,-0.00517038500000000,-0.00656741750000000,-0.00413256333333333,-0.0102880500000000,-0.00121275833333333,-0.00306071583333333,-0.00270940083333333,-0.00575980500000000,-0.00395225333333333,-0.00883093666666667,-0.00640212833333333,-0.0132689483333333,-0.00103013000000000,-0.00711574666666667,-0.00994204250000000,-0.0199892300000000,-0.00495133333333333,-0.00134110500000000,-0.00294384250000000,-0.00706383500000000,-0.00268476333333333,-0.0576522175000000,-0.00498249416666667,-0.00463285583333333,-0.00577282500000000,-0.00103157166666667,-0.0117600341666667,-0.00120604083333333,-0.00269287166666667,-0.00102005833333333,-0.0552463858333333,-0.00528291500000000,-0.00405974166666667,-0.00628113583333333,0.000652742500000000,-0.0146826758333333,-0.00172142250000000,0.000880473333333333,-8.13558333333333e-05,-0.00775991250000000};
                // normalization parameters for each region
                float b_old[379] = {-0.0127491033333333,-0.00423616916666667,-0.00693808916666667,-0.0192015400000000,-0.0141342941666667,-0.0170751758333333,-0.00323064333333333,-0.00712867666666667,-0.00696357250000000,-0.00511082750000000,-0.00399446500000000,-0.00162936083333333,-0.0125255225000000,0.000892350833333333,-0.00345203750000000,-0.00563618166666667,-0.00743483666666667,-0.00417684000000000,-0.00704786250000000,-0.00608268000000000,-0.00194381333333333,-0.00207084250000000,-0.00203272916666667,-0.000223961666666667,-0.00609640833333333,0.00483195916666667,3.00166666666666e-06,-0.00257795000000000,0.00115106500000000,0.00610350500000000,7.16333333333334e-06,0.00218943166666667,0.00642470666666667,0.00351119250000000,0.00510465833333333,0.00101464166666667,-0.00216545666666667,-0.00387289666666667,-0.000118760000000000,0.000384150833333333,0.000147728333333333,-0.00215620166666667,-0.00307712750000000,-0.00307005083333333,-0.00590053333333333,-0.00475679666666667,-0.00381207916666667,-0.00655079250000000,-0.00412691166666667,-0.00981231750000000,-0.00655823083333333,-0.00990049166666667,-0.00424837000000000,0.000526334166666667,-0.000985988333333333,-0.00368209166666667,-0.00116140000000000,0.000426270833333333,-0.000570095833333333,-0.00269987250000000,0.00383888916666667,0.00125463916666667,-0.000479164166666667,0.00198327333333333,0.00346832250000000,0.00292580833333333,0.00476076333333333,0.00459313166666667,0.000358307500000000,0.00636275666666667,0.00514540583333333,0.00523009250000000,-0.00125120250000000,0.000799418333333333,0.00134007500000000,0.00417445250000000,0.00149398750000000,-0.00901695333333333,-0.00229982166666667,-0.000983602500000000,-0.00391603916666667,-0.00745765166666667,-0.00455852333333333,-0.00778180833333333,1.99575000000000e-05,-0.00341958416666667,0.00711167916666667,0.00576907250000000,0.00293310083333333,0.00404899250000000,0.00139994083333333,-0.00103056333333333,0.00205062416666667,0.00311168500000000,-0.00404026666666667,-0.00592452916666667,0.00204146083333333,0.00481788666666667,-0.00171801916666667,-0.00318577750000000,0.000539700833333333,0.00189802916666667,0.000591975000000000,-0.00291871666666667,-0.00305066833333333,-0.000904960833333333,0.00155941833333333,-0.00236333916666667,-0.00224454583333333,0.00112914750000000,0.000953092500000000,0.00220989250000000,-0.000259302500000000,-0.000316035000000000,0.000664457500000000,-0.00482794583333333,-0.0113679483333333,0.00172640416666667,-0.000555599166666667,-0.000805580833333333,-0.00139773416666667,0.00127089000000000,0.00156711000000000,-0.00147980166666667,-0.00131369083333333,0.00153299833333333,-0.000120919166666667,0.00230405583333333,0.00132433250000000,0.00352820250000000,0.00328530333333333,0.0101075091666667,-0.00190666000000000,0.00441255583333333,0.000435692500000000,-0.00166781333333333,-0.00837684666666667,-0.00773250833333333,-0.00468703166666667,-0.00563043333333333,-0.00376537000000000,-0.00730610583333333,-0.0130715166666667,-0.00557068750000000,-0.00354334166666667,-0.00924092000000000,-0.00539229083333333,-0.0160644941666667,-0.00627113750000000,0.000539764166666667,0.000116330000000000,-0.00437938000000000,-0.00267223166666667,-0.00149006916666667,0.00138454916666667,-0.00387393500000000,-0.00556845750000000,-0.00828047166666667,-0.00724820666666667,-0.00221066916666667,0.00669863083333333,0.00173144833333333,-0.00238003000000000,0.00344941750000000,0.00382308750000000,0.00166150750000000,-0.000491025833333333,0.00220296416666667,-0.000587458333333333,0.00242705666666667,-0.000671671666666667,0.000921503333333333,0.000712058333333333,-0.000843085000000000,-0.00394298750000000,0.00731971416666667,0.00562133916666667,0.000220499166666667,-0.00107310250000000,0.00180522916666667,-0.0128085225000000,-0.00284355166666667,-0.00549121000000000,-0.0168757250000000,-0.0126110808333333,-0.0164709816666667,-0.00209039666666667,-0.0109291066666667,-0.00982754583333333,-0.00477670500000000,-0.00246149000000000,6.86400000000000e-05,-0.0122703708333333,0.00188976583333333,-0.00163406166666667,-0.00501329583333333,-0.00837608583333333,-0.00292528750000000,-0.00497000000000000,-0.00390848500000000,-0.00361591166666667,-0.00207562666666667,-0.00195883416666667,-0.000255772500000000,0.000102730000000000,0.00459359666666667,0.000197609166666667,-0.00102579166666667,0.00218534083333333,0.00371827666666667,0.00178773000000000,0.00254521583333333,0.00484219333333333,0.00344504333333333,0.00494256166666667,0.000383162500000000,-0.00227357083333333,-0.00387956916666667,-0.00112874750000000,-0.000689509166666667,-0.000934042500000000,-0.00386068416666667,-0.00380912750000000,-0.00346816583333333,-0.00551153333333333,-0.00541183583333333,-0.00404711250000000,-0.00450341916666667,-0.00379112083333333,-0.00885549500000000,-0.00826772500000000,-0.0102048508333333,-0.00672982416666667,-0.000148650000000000,-0.00101832666666667,-0.00282455083333333,0.000981301666666667,0.00188302666666667,-0.000397391666666667,-0.00217559666666667,0.00387339416666667,0.00183665583333333,-0.000473489166666667,0.00299008416666667,0.00389108416666667,0.00210154416666667,0.00561740333333333,0.00455129083333333,0.00102152000000000,0.00746882750000000,0.00652763333333333,0.00357186833333333,-0.000961076666666667,0.00106191916666667,0.00207869416666667,0.00476111916666667,0.00185498000000000,-0.00852459083333333,-0.000366460000000000,-0.00269433916666667,-0.000367614166666667,-0.00781396000000000,-0.00506720166666667,-0.00942865916666667,-0.00253627333333333,-0.00645129666666667,0.00468729750000000,0.00647645500000000,0.00449891500000000,0.00546691583333333,-0.00152215250000000,-0.00139003583333333,0.000539448333333333,0.00377045666666667,-0.00477911333333333,-0.00692656416666667,0.00163592250000000,0.00372001916666667,-0.00234267000000000,-0.00450356083333333,-0.000426252500000000,-0.000935375000000000,0.000538911666666667,-0.00140079750000000,-0.00306439333333333,-0.00186524250000000,0.000422480833333333,-0.00400739250000000,-0.00217679083333333,-0.00173223916666667,0.000119601666666667,0.000744389166666667,-0.000466585833333333,-0.00104640750000000,-0.00161728000000000,-0.00676775250000000,-0.0104170375000000,0.00195996666666667,0.00194405750000000,0.000210285000000000,-2.34300000000000e-05,0.00112546583333333,0.00167144583333333,-0.00383098500000000,-0.00127078083333333,0.00163178833333333,0.000976291666666667,0.00246910583333333,0.00111628500000000,0.00306915333333333,0.00351383666666667,0.0101248850000000,-0.00124539000000000,0.00414075250000000,0.00203580666666667,-0.00164575750000000,-0.00874299833333333,-0.00998759416666667,-1.21391666666667e-05,-0.00389190750000000,-0.000952643333333334,-0.00464539333333333,-0.0101699141666667,-0.00599946916666667,-0.00365333583333333,-0.00657917250000000,-0.00710916333333333,-0.0144727075000000,-0.00629214750000000,0.000906763333333333,6.06883333333333e-05,-0.00466602666666667,-0.00158147416666667,-0.00117202416666667,0.00279109666666667,-0.00362337750000000,-0.00405712083333333,-0.00502669416666667,-0.00469704916666667,-0.000228719166666667,0.00614735750000000,0.00258339333333333,-0.00199113333333333,0.000862420833333333,0.000986107500000000,-0.00112716250000000,-0.00143542500000000,0.000362182500000000,-0.00212245500000000,0.00291865416666667,-0.00338183416666667,0.00310177916666667,-0.000729680833333333,-0.000268673333333333,-0.00487705666666667,0.00634932916666667,0.00757651416666667,0.000659699166666667,0.000782888333333333,0.00204377166666667,-0.00741839083333333,0.000851010000000000,0.00133849750000000,0.000283717500000000,0.000705759166666667,-0.00206706166666667,0.00176285250000000,-0.000977167500000000,0.00211537833333333,-0.00792832750000000,0.000611960000000000,0.00142314750000000,-5.44141666666667e-05,0.00297483500000000,-0.00381691166666667,0.00112966000000000,0.00256948833333333,0.00331060000000000,0.000316330833333333};

                float a_new = -0.006;
                float b_new =  0.001;
                
                // Input PPC (Module 1)
                for (i=0; i < 10; i++) {
                    j=PPC_regs[i];
                    
                    // update noise process
                    I_noiseA[j] = MW_noise(I_noiseA[j]);
                    I_noiseB[j] = MW_noise(I_noiseB[j]);
                    
                    // Normalize input to MW
                    //float tmp_inp = (I_E[j] - PPC_amps[i]) / PPC_SD[i] * MW_noise_SD * frac_BNM + PPC_amps[i];
                    //float tmp_inp = I_E[j] * MW_noise_SD / median_std_indiBNMs * fac;
                    //float tmp_inp = (I_E[j] - prctile70_amp[j]) / prctile10_std[j] * MW_noise_SD * sqrtf(lower_corr);
                    float tmp_inp = (b_new - a_new) * (I_E[j] - a_old[j]) / (b_old[j] - a_old[j]) + a_new;

                    I_PPC_A[j] =  I_noiseA[j] + tmp_inp;
                    I_PPC_B[j] =  I_noiseB[j] + tmp_inp;
                  
                    amp_PPC[j] += I_E[j];
                    amp_PPCs[j] += I_PPC_A[j];
                }
                
                // Input PFC (Module 2)
                for (i=0; i < 9; i++) {
                    j=PFC_regs[i];
                    
                    // update noise process
                    I_noiseA[j] = MW_noise(I_noiseA[j]);
                    I_noiseB[j] = MW_noise(I_noiseB[j]);
                    
                    // Normalize input to MW
                    //float tmp_inp = (I_E[j] - PPC_amps[i]) / PPC_SD[i] * MW_noise_SD * frac_BNM + PPC_amps[i];
                    //float tmp_inp = I_E[j] * MW_noise_SD / median_std_indiBNMs * fac;
                    //float tmp_inp = (I_E[j] - prctile70_amp[j]) / prctile10_std[j] * MW_noise_SD * sqrtf(lower_corr);
                    float tmp_inp = (b_new - a_new) * (I_E[j] - a_old[j]) / (b_old[j] - a_old[j]) + a_new;
                    
                    I_PFC_A[j] =  I_noiseA[j] + tmp_inp;
                    I_PFC_B[j] =  I_noiseB[j] + tmp_inp;
                    
                    amp_PFC[j] += I_E[j];
                    amp_PFCs[j] += I_PFC_A[j];
                }
            
        
                int i_PPC, i_PFC;
                int i_ppcpfc = 0;
                for (i_PPC = 0; i_PPC < 10; i_PPC++) {
                    for (i_PFC = 0; i_PFC < 9; i_PFC++) {
                        int r_PPC = PPC_regs[i_PPC];
                        int r_PFC = PFC_regs[i_PFC];

                        // Integrate MurrayWang PPC and PFC
    //                    float contrast = 5.0;
                        float contrast = inp_contrast;
                        float I_appA = 0.0118 * (1 + contrast / 100);
                        float I_appB = 0.0118 * (1 - contrast / 100);

                        // sum input
                        float I_A1 = fI_A1(S_A1[i_ppcpfc], S_A2[i_ppcpfc], S_B1[i_ppcpfc], S_B2[i_ppcpfc], I_PPC_A[r_PPC], I_appA);
                        float I_B1 = fI_B1(S_A1[i_ppcpfc], S_A2[i_ppcpfc], S_B1[i_ppcpfc], S_B2[i_ppcpfc], I_PPC_B[r_PPC], I_appB);
                        float I_A2 = fI_A2(S_A1[i_ppcpfc], S_A2[i_ppcpfc], S_B1[i_ppcpfc], S_B2[i_ppcpfc], I_PFC_A[r_PFC]);
                        float I_B2 = fI_B2(S_A1[i_ppcpfc], S_A2[i_ppcpfc], S_B1[i_ppcpfc], S_B2[i_ppcpfc], I_PFC_B[r_PFC]);

                        // compute firing rates
                        r_A1[i_ppcpfc].ts[s_ts] = MW_r(I_A1);
                        r_B1[i_ppcpfc].ts[s_ts] = MW_r(I_B1);
                        r_A2[i_ppcpfc].ts[s_ts] = MW_r(I_A2);
                        r_B2[i_ppcpfc].ts[s_ts] = MW_r(I_B2);

                        // compute synaptic gating
                        S_A1[i_ppcpfc] = S_AB(S_A1[i_ppcpfc], r_A1[i_ppcpfc].ts[s_ts]);
                        S_B1[i_ppcpfc] = S_AB(S_B1[i_ppcpfc], r_B1[i_ppcpfc].ts[s_ts]);
                        S_A2[i_ppcpfc] = S_AB(S_A2[i_ppcpfc], r_A2[i_ppcpfc].ts[s_ts]);
                        S_B2[i_ppcpfc] = S_AB(S_B2[i_ppcpfc], r_B2[i_ppcpfc].ts[s_ts]);

                        i_ppcpfc++;
                    }
                }
                s_ts++;
            }
        } // Simulation loop
    
    //    int PPC_regs[10] = { (117-1), (95-1), (144-1), (145-1), (46-1), (29-1), (45-1), (15-1), (149-1), (151-1) };
    //    int PFC_regs[9] = { (85-1), (86-1), (83-1), (73-1), (97-1), (98-1), (67-1), (26-1), (63-1) };
    //    char desc_file[100];memset(desc_file, 0, 100*sizeof(char));
    //    snprintf(desc_file,sizeof(desc_file),"DMoutput_seed%d.txt", new_seed);
    //    FILE *tmp_file = fopen(desc_file, "w");
    //    int i;
    //    for (i=0; i < burn_in_ts; i++) {
    //        for (k=0; k < 90; k++) {
    //            fprintf(tmp_file, "%.9f %.9f %.9f %.9f ", r_A1[k].ts[i], r_A2[k].ts[i], r_B1[k].ts[i], r_B2[k].ts[i]);
    //        }
    //        fprintf(tmp_file, "\n");
    //    }
        // write out mean
    //    for (k=0; k < 10; k++) {
    //        j=PPC_regs[k];
    //        float m = mean(&tmp_out[j].ts[0],s_ts-1);
    //        fprintf(tmp_file, "%.9f ", m);
    //    }
    //    for (k=0; k < 9; k++) {
    //        j=PFC_regs[k];
    //        float m = mean(&tmp_out[j].ts[0],s_ts-1);
    //        fprintf(tmp_file, "%.9f ", m);
    //    }
    //    fprintf(tmp_file, "\n");
    //    // write out std
    //    for (k=0; k < 10; k++) {
    //        j=PPC_regs[k];
    //        float s = std(&tmp_out[j].ts[0],s_ts-1);
    //        fprintf(tmp_file, "%.9f ", s);
    //    }
    //    for (k=0; k < 9; k++) {
    //        j=PFC_regs[k];
    //        float s = std(&tmp_out[j].ts[0],s_ts-1);
    //        fprintf(tmp_file, "%.9f ", s);
    //    }
    //    fprintf(tmp_file, "\n");
    //    fclose(tmp_file);
    


        /* Evaluate DM experiment */
        fprintf(tmp_corrdesc, "%.2f %.2f %.1f %d %d %.6f %.6f %.6f %.6f\n", (float)new_seed, (float)new_seed, (float)new_seed, new_seed, new_seed, (float)new_seed, (float)new_seed, (float)new_seed, (float)new_seed);
        
        int i_PPC, i_PFC;
        int i_ppcpfc = 0;
        float *correct_response = (float *)_mm_malloc(90 * sizeof(float),16);
        int *dec_time_A = (int *)_mm_malloc(90 * sizeof(float),16);
        int *dec_time_B = (int *)_mm_malloc(90 * sizeof(float),16);
        for (i_PPC = 0; i_PPC < 10; i_PPC++) {
            for (i_PFC = 0; i_PFC < 9; i_PFC++) {

                float mean_A2 = mean(&r_A2[i_ppcpfc].ts[4800],150);
                float mean_B2 = mean(&r_B2[i_ppcpfc].ts[4800],150);

                if (mean_A2 > 30 && mean_B2 < 30) correct_response[i_ppcpfc] = 1.0;
                else correct_response[i_ppcpfc] = 0.0;

                dec_time_A[i_ppcpfc] = dec_time(&r_A1[i_ppcpfc].ts[0], s_ts);
                dec_time_B[i_ppcpfc] = dec_time(&r_B1[i_ppcpfc].ts[0], s_ts);
                
                int r_PPC = PPC_regs[i_PPC];
                int r_PFC = PFC_regs[i_PFC];

                fprintf(tmp_corrdesc, "%.2f %.2f %.1f %d %d %.6f %.6f %.6f %.6f\n", mean_A2, mean_B2, correct_response[i_ppcpfc], dec_time_A[i_ppcpfc], dec_time_B[i_ppcpfc], amp_PPC[r_PPC]/(float)s_ts, amp_PPCs[r_PPC]/(float)s_ts, amp_PFC[r_PFC]/(float)s_ts, amp_PFCs[r_PFC]/(float)s_ts);

                i_ppcpfc++;
            }
        }
    }
    fclose(tmp_corrdesc);

    printf("TVBii FIC tuning finished. Execution took %.2f s\n", (float)(time(NULL) - start));
    
    return 0;
}
