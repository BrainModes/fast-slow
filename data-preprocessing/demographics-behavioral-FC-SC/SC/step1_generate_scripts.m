clear
close all
clc


subjects =   load('subject_list.txt');
cd('scripts')
delete('*')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Convert labels to volumes
% wb_command -label-to-volume-mapping MMP_CORTEX_LEFT.label.gii 116726.L.white_MSMAll.32k_fs_LR.surf.gii T1w_acpc_dc_restore_brain.nii.gz CORTEX_LEFT_vol.nii -ribbon-constrained 116726.L.white_MSMAll.32k_fs_LR.surf.gii 116726.L.pial_MSMAll.32k_fs_LR.surf.gii
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initstring = ['#!/bin/bash -x\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --time=00:10:00\n#SBATCH --partition=batch\n'];

connectome_folder             = '/homeb/slns/slns019/HCP_MMP_Connectomes/SC/';
hcp_data_folder               = '/homec/hbu23/hbu231/_data/hcp_data/hcp/';
label_folder                  = '/homeb/slns/slns019/HCP_MMP_Connectomes/SC/label_to_volume_mapping/';
surf_folder                   = '/homeb/slns/slns019/hcp_data/';
mmp_left_labels               = [connectome_folder 'MMP_CORTEX_LEFT.label.gii'];
mmp_right_labels              = [connectome_folder 'MMP_CORTEX_RIGHT.label.gii'];

break_counter = 0;
preproc_patch = 0;
fileID_batch = fopen('step1_label2volume','w');
for ii = 1:length(subjects)
    if break_counter == 0
        command     = initstring;        
    end
    break_counter = break_counter + 2;       

    white_left      = [surf_folder num2str(subjects(ii)) '.L.white_MSMAll.32k_fs_LR.surf.gii']; 
    white_right     = [surf_folder num2str(subjects(ii)) '.R.white_MSMAll.32k_fs_LR.surf.gii']; 
    pial_left       = [surf_folder num2str(subjects(ii)) '.L.pial_MSMAll.32k_fs_LR.surf.gii']; 
    pial_right      = [surf_folder num2str(subjects(ii)) '.R.pial_MSMAll.32k_fs_LR.surf.gii']; 
    T1w             = [hcp_data_folder num2str(subjects(ii)) '/T1w/T1w_acpc_dc_restore_brain.nii.gz'];
    output_left     = [label_folder num2str(subjects(ii)) '.LEFT_vol.nii'];
    output_right    = [label_folder num2str(subjects(ii)) '.RIGHT_vol.nii'];
    
    command         = [command 'srun -n 1 --exclusive wb_command -label-to-volume-mapping ' mmp_left_labels  ' ' white_left  ' ' T1w ' ' output_left  ' -ribbon-constrained ' white_left  ' ' pial_left  ' &\n'];
    command         = [command 'srun -n 1 --exclusive wb_command -label-to-volume-mapping ' mmp_right_labels ' ' white_right ' ' T1w ' ' output_right ' -ribbon-constrained ' white_right ' ' pial_right ' &\n'];

    if break_counter >= 24 || ii == length(subjects)
        preproc_patch = preproc_patch + 1;
        finalstring = [command '\nwait\n'];
        fileID = fopen(['step1_label2volume_' num2str(preproc_patch)],'w');
        fprintf(fileID,finalstring);
        fclose(fileID);        
        fprintf(fileID_batch, ['sbatch step1_label2volume_' num2str(preproc_patch) '\n']);
        break_counter = 0;
    end
end
fclose(fileID_batch);

