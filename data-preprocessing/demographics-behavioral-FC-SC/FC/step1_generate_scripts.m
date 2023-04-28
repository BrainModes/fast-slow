% This file generates SLURM scripts for parallel processing on HPC
clear
close all
clc

subjects = load('subject_list.txt');
cd('scripts')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Export region-average time series for MMP
% wb_command -cifti-parcellate rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii 181232.aparc.32k_fs_LR.dlabel.nii COLUMN output_aparc.ptseries.nii
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initstring = ['#!/bin/bash -x\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --time=00:10:00\n#SBATCH --partition=batch\n'];

connectome_folder             = '/homeb/slns/slns019/HCP_MMP_Connectomes/FC/extracted_ptseries_nii/';
hcp_data_folder               = '/homec/hbu23/hbu231/_data/hcp_data/hcp/';
label_folder                  = '/homeb/slns/slns019/HCP_MMP_Connectomes/FC/';

break_counter = 0;
preproc_patch = 0;
fileID_batch = fopen('step1_cifti_parcellate','w');
for ii = 1:length(subjects)
    if break_counter == 0
        command     = initstring;        
    end
    break_counter = break_counter + 8;       

    input_file1     = [hcp_data_folder num2str(subjects(ii)) '/fMRI/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii']; 
    input_file2     = [hcp_data_folder num2str(subjects(ii)) '/fMRI/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii']; 
    input_file3     = [hcp_data_folder num2str(subjects(ii)) '/fMRI/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii']; 
    input_file4     = [hcp_data_folder num2str(subjects(ii)) '/fMRI/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii']; 
    label_MMP       = [label_folder    'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii']; 
    label_subcort   = [label_folder    'Atlas_ROIs.2.dlabel.nii']; 
    output_file1_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST1_LR_MMP.ptseries.nii']; 
    output_file1_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST1_LR_subcort.ptseries.nii']; 
    output_file2_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST1_RL_MMP.ptseries.nii']; 
    output_file2_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST1_RL_subcort.ptseries.nii']; 
    output_file3_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST2_LR_MMP.ptseries.nii']; 
    output_file3_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST2_LR_subcort.ptseries.nii']; 
    output_file4_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST2_RL_MMP.ptseries.nii']; 
    output_file4_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST2_RL_subcort.ptseries.nii']; 

    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file1 ' ' label_MMP ' COLUMN ' output_file1_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file1 ' ' label_subcort ' COLUMN ' output_file1_subcort ' &\n'];

    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file2 ' ' label_MMP ' COLUMN ' output_file2_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file2 ' ' label_subcort ' COLUMN ' output_file2_subcort ' &\n'];

    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file3 ' ' label_MMP ' COLUMN ' output_file3_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file3 ' ' label_subcort ' COLUMN ' output_file3_subcort ' &\n'];

    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file4 ' ' label_MMP ' COLUMN ' output_file4_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-parcellate  ' input_file4 ' ' label_subcort ' COLUMN ' output_file4_subcort ' &\n'];
    
    if break_counter == 24 || ii == length(subjects)
        preproc_patch = preproc_patch + 1;
        finalstring = [command '\nwait\n'];
        fileID = fopen(['step1_cifti_parcellate_' num2str(preproc_patch)],'w');
        fprintf(fileID,finalstring);
        fclose(fileID);        
        fprintf(fileID_batch, ['sbatch step1_cifti_parcellate_' num2str(preproc_patch) '\n']);
        break_counter = 0;
    end
end
fclose(fileID_batch);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Convert to text file
% wb_command -cifti-convert -to-text output_aparc.ptseries.nii output_aparc.ptseries.txt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initstring = ['#!/bin/bash -x\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --time=00:10:00\n#SBATCH --partition=batch\n'];

connectome_folder             = '/homeb/slns/slns019/HCP_MMP_Connectomes/FC/extracted_ptseries_nii/';
connectome_folder2            = '/homeb/slns/slns019/HCP_MMP_Connectomes/FC/extracted_ptseries_txt/';
hcp_data_folder               = '/homec/hbu23/hbu231/_data/hcp_data/hcp/';
label_folder                  = '/homeb/slns/slns019/HCP_MMP_Connectomes/FC/';

break_counter = 0;
preproc_patch = 0;
fileID_batch = fopen('step2_to_txt','w');
for ii = 1:length(subjects)
    if break_counter == 0
        command     = initstring;        
    end
    break_counter = break_counter + 8;     
    
    input_file1_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST1_LR_MMP.ptseries.nii']; 
    input_file1_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST1_LR_subcort.ptseries.nii']; 
    input_file2_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST1_RL_MMP.ptseries.nii']; 
    input_file2_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST1_RL_subcort.ptseries.nii']; 
    input_file3_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST2_LR_MMP.ptseries.nii']; 
    input_file3_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST2_LR_subcort.ptseries.nii']; 
    input_file4_MMP        = [connectome_folder num2str(subjects(ii)) '_FC_REST2_RL_MMP.ptseries.nii']; 
    input_file4_subcort    = [connectome_folder num2str(subjects(ii)) '_FC_REST2_RL_subcort.ptseries.nii']; 
    
    output_file1_MMP        = [connectome_folder2 num2str(subjects(ii)) '_FC_REST1_LR_MMP.ptseries.txt']; 
    output_file1_subcort    = [connectome_folder2 num2str(subjects(ii)) '_FC_REST1_LR_subcort.ptseries.txt']; 
    output_file2_MMP        = [connectome_folder2 num2str(subjects(ii)) '_FC_REST1_RL_MMP.ptseries.txt']; 
    output_file2_subcort    = [connectome_folder2 num2str(subjects(ii)) '_FC_REST1_RL_subcort.ptseries.txt']; 
    output_file3_MMP        = [connectome_folder2 num2str(subjects(ii)) '_FC_REST2_LR_MMP.ptseries.txt']; 
    output_file3_subcort    = [connectome_folder2 num2str(subjects(ii)) '_FC_REST2_LR_subcort.ptseries.txt']; 
    output_file4_MMP        = [connectome_folder2 num2str(subjects(ii)) '_FC_REST2_RL_MMP.ptseries.txt']; 
    output_file4_subcort    = [connectome_folder2 num2str(subjects(ii)) '_FC_REST2_RL_subcort.ptseries.txt']; 
    
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file1_MMP ' ' output_file1_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file1_subcort  ' ' output_file1_subcort ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file2_MMP ' ' output_file2_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file2_subcort  ' ' output_file2_subcort ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file3_MMP ' ' output_file3_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file3_subcort  ' ' output_file3_subcort ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file4_MMP ' ' output_file4_MMP ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -cifti-convert -to-text ' input_file4_subcort  ' ' output_file4_subcort ' &\n'];
    
    if break_counter == 24 || ii == length(subjects)
        preproc_patch = preproc_patch + 1;
        finalstring = [command '\nwait\n'];
        fileID = fopen(['step2_to_txt_' num2str(preproc_patch)],'w');
        fprintf(fileID,finalstring);
        fclose(fileID);        
        fprintf(fileID_batch, ['sbatch step2_to_txt_' num2str(preproc_patch) '\n']);
        break_counter = 0;
    end
end
fclose(fileID_batch);





