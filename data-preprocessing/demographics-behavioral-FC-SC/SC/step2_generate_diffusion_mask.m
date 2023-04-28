function step2_generate_diffusion_mask(sub_id)

% Generate Diffusion Mask

% Step 1: Clean aparc+aseg.nii.gz such that only subcortical structures
% remain

% Step 2: Merge cortical and subcortical volumes


% Add NIFTI tools or FreeSurfer Matlab folder to Matlab path
addpath(genpath('/homec/hbu23/hbu231/Octave_include/NIFTI'));

% Include the following subcortical structures

subcortical_regions = [8;10;11;12;13;16;17;18;26;28;47;49;50;51;52;53;54;58;60];

% FreeSurfer Color LUT
% https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
% old_id region_label                 new_id
% 8   Left-Cerebellum-Cortex          --> 361                                     
% 10  Left-Thalamus-Proper            --> 362       
% 11  Left-Caudate                    --> 363        
% 12  Left-Putamen                    --> 364        
% 13  Left-Pallidum                   --> 365                                  
% 16  Brain-Stem                      --> 379        
% 17  Left-Hippocampus                --> 366        
% 18  Left-Amygdala                   --> 367        
% 26  Left-Accumbens-area             --> 368                        
% 28  Left-VentralDC                  --> 369                                            
% 47  Right-Cerebellum-Cortex         --> 370                                 
% 49  Right-Thalamus-Proper           --> 371        
% 50  Right-Caudate                   --> 372        
% 51  Right-Putamen                   --> 373        
% 52  Right-Pallidum                  --> 374        
% 53  Right-Hippocampus               --> 375        
% 54  Right-Amygdala                  --> 376                                    
% 58  Right-Accumbens-area            --> 377                         
% 60  Right-VentralDC                 --> 378


% Load cortical/subcortical segmentation
aparc_file = ['/homec/hbu23/hbu231/_data/hcp_data/hcp/' num2str(sub_id) '/T1w/aparc+aseg.nii'];
aparcaseg  = load_untouch_nii(aparc_file,[],[],[],[],[],[]);


% Set all voxel labels to zero that do not belong to a subcortical
% structure
aparcaseg.img(~ismember(aparcaseg.img, subcortical_regions)) = 0;


% Assign new ids to subcortical structures in order to not interfere with
% HCP MMP1.0 Atlas (360 regions)
% Note that brainstem is moved to the end

subcortical_regions = [8;10;11;12;13;17;18;26;28;47;49;50;51;52;53;54;58;60;16];

idx_subcort = 361;
for ii = subcortical_regions'
    aparcaseg.img(aparcaseg.img == ii) = idx_subcort;
    idx_subcort = idx_subcort + 1;
end

%save_untouch_nii(aparcaseg, 'aparc+aseg_MMP.nii.gz');
subcortex           = aparcaseg;

% Load three segmentation volumes
%subcortex           = load_untouch_nii('aparc+aseg_MMP.nii');
cortex_left_file    = ['/homeb/slns/slns019/HCP_MMP_Connectomes/SC/label_to_volume_mapping/' num2str(sub_id) '.LEFT_vol.nii'];
cortex_left         = load_untouch_nii(cortex_left_file,[],[],[],[],[],[]);
cortex_right_file   = ['/homeb/slns/slns019/HCP_MMP_Connectomes/SC/label_to_volume_mapping/' num2str(sub_id) '.RIGHT_vol.nii'];
cortex_right        = load_untouch_nii(cortex_right_file,[],[],[],[],[],[]);


% Some border voxels of the subcortical segmentation (Hippocampus and Cerebellum)
% overlap with MMP parcellation, also some right/and left MMP cortex voxels overlap
% -> since they cannot be assigned to a single region: clean these voxels
% out of all masks
sc  = subcortex.img;
lc  = cortex_left.img;
rc  = cortex_right.img;

lc(lc>0) = 1;
rc(rc>0) = 1;
sc(sc>0) = 1;
overlap  = lc+rc+sc;
overlap  = find(overlap > 1);

subcortex.img(overlap)      = 0;
cortex_left.img(overlap)    = 0;
cortex_right.img(overlap)   = 0;

% Merge masks
cortex_left.img     = cortex_left.img + cortex_right.img + subcortex.img;

% Save result
output_diffusion_mask_file = ['/homeb/slns/slns019/HCP_MMP_Connectomes/SC/diffusion_masks/' num2str(sub_id) '.diffmask_MMP.nii'];
save_untouch_nii(cortex_left, output_diffusion_mask_file);

end

%%
% clear
% clc
% close all
% cd('/Users/michael/Desktop/HCP_project/MMP_Connectomes/SC/scripts/')
% subjects            =   load('/Users/michael/Desktop/HCP_project/subject_list.txt');
% 
% initstring = ['#!/bin/bash -x\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --time=00:10:00\n#SBATCH --partition=batch\nmodule load GCC/7.2.0  ParaStationMPI/5.2.0-1\nmodule load Octave/4.2.1\n\n'];
% batch_line = 'srun -n 1 octave --eval "step2_generate_diffusion_mask(';
% fileID_start = fopen(['step2_diffmask'],'w');
% 
% 
% break_counter = 0;
% preproc_patch = 0;
% for ii = 1:length(subjects)
%     if break_counter == 0
%         command     = initstring;        
%     end
%     break_counter = break_counter + 1;   
%    
%     command      = [command batch_line num2str(subjects(ii))  ')" &\n'];
%     display(['gunzip /homec/hbu23/hbu231/_data/hcp_data/hcp/' num2str(subjects(ii)) '/T1w/aparc+aseg.nii.gz'])
% 
%     if break_counter == 24 || ii == length(subjects)
%         preproc_patch = preproc_patch + 1;
%         finalstring = [command '\nwait\n'];
%         fileID = fopen(['step2_diffmask_' num2str(preproc_patch)],'w');
%         fprintf(fileID,finalstring);
%         fclose(fileID);
%         fprintf(fileID_start, ['sbatch step2_diffmask_' num2str(preproc_patch) '\n']);
%         break_counter = 0;
%     end
% end
% 
% fclose(fileID_start)
% 
% 
% 
