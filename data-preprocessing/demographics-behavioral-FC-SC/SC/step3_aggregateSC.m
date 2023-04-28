% This script aggregates tractography output
% Tractography was performed according to this tutorial:
% https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/ismrm_hcp_tutorial.html

clear
close all
clc


subjects  =   load('subject_list.txt');
cd('scripts_step3')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Map streamlines to the parcellated image to produce a connectome
% tck2connectome 10M_SIFT.tck nodes_fixSGM.mif connectome.csv
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initstring = ['#!/bin/bash -x\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --time=00:10:00\n#SBATCH --partition=batch\nmodule load GCC/5.4.0\nmodule load MRtrix/0.3.15-Python-2.7.14\n\n'];
initstring = ['#!/bin/bash -x\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --time=00:20:00\n#SBATCH --partition=batch\nmodule use /usr/local/software/jureca/OtherStages\nmodule load Stages/2017a\nmodule load Stages/2016b  GCC/5.4.0\nmodule load MRtrix/0.3.15-Python-2.7.12\n\n'];

connectome_folder             = '/homeb/slns/slns019/HCP_MMP_Connectomes/SC/SC/';
sift_weights_folder           = '/work/hbu23/hbu231/HCP_tracks_sift2/';
tracks_folder                 = '/work/hbu23/hbu231/archHCPtcks/';  
parc_folder                   = '/homeb/slns/slns019/HCP_MMP_Connectomes/SC/diffusion_masks/';


break_counter = 0;
preproc_patch = 0;
fileID_batch = fopen('step3_tck2connectome','w');
for ii = 1:length(subjects)
    if break_counter == 0
        command     = initstring;        
    end
    break_counter = break_counter + 2;       

    tck_file             = [tracks_folder            num2str(subjects(ii)) '_25M_tracks.tck'];
    siftweights_file     = [sift_weights_folder      num2str(subjects(ii)) '/' num2str(subjects(ii)) '_25M_tracks_weights.txt'];
    parc                 = [parc_folder              num2str(subjects(ii)) '.diffmask_MMP.nii'];
    connectome           = [connectome_folder        num2str(subjects(ii)) '_SC.csv'];
    connectome_len       = [connectome_folder        num2str(subjects(ii)) '_SC_len.csv'];
    
    command         = [command 'srun -n 1 --exclusive -c 1 tck2connectome -tck_weights_in ' siftweights_file ' ' tck_file ' ' parc ' ' connectome ' -nthreads 1 -force &\n'];
    command         = [command 'srun -n 1 --exclusive -c 1 tck2connectome -scale_length -stat_edge mean -tck_weights_in ' siftweights_file ' ' tck_file ' ' parc ' ' connectome_len ' -nthreads 1 -force &\n'];
    
    if break_counter >= 48 || ii == length(subjects)
        preproc_patch = preproc_patch + 1;
        finalstring = [command '\nwait\n'];
        fileID = fopen(['step3_tck2connectome_' num2str(preproc_patch)],'w');
        fprintf(fileID,finalstring);
        fclose(fileID);        
        fprintf(fileID_batch, ['sbatch step3_tck2connectome_' num2str(preproc_patch) '\n']);
        break_counter = 0;
    end
end
fclose(fileID_batch);





