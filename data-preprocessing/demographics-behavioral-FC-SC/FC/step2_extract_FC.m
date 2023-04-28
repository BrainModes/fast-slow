function step2_extract_FC(sub_ii)

%%%%%%%%%%%%%%%%%%%%%%%
% Mapping between atlases
%%%%%%%%%%%%%%%%%%%%%%%
% 
% ROIs 1 to 360 from MMP
% ROIs 361 to 379 from Atlas_ROIs.2.dlabel.nii

%{

************************
Sorting in SC
************************
% 1. 8   Left-Cerebellum-Cortex          --> 361                                     
% 2. 10  Left-Thalamus-Proper            --> 362       
% 3. 11  Left-Caudate                    --> 363        
% 4. 12  Left-Putamen                    --> 364        
% 5. 13  Left-Pallidum                   --> 365                                  
% 6. 16  Brain-Stem                      --> 379        
% 7. 17  Left-Hippocampus                --> 366        
% 8. 18  Left-Amygdala                   --> 367        
% 9. 26  Left-Accumbens-area             --> 368                        
% 10.28  Left-VentralDC                  --> 369                                            
% 11.47  Right-Cerebellum-Cortex         --> 370                                 
% 12.49  Right-Thalamus-Proper           --> 371        
% 13.50  Right-Caudate                   --> 372        
% 14.51  Right-Putamen                   --> 373        
% 15.52  Right-Pallidum                  --> 374        
% 16.53  Right-Hippocampus               --> 375        
% 17.54  Right-Amygdala                  --> 376                                    
% 18.58  Right-Accumbens-area            --> 377                         
% 19.60  Right-VentralDC                 --> 378


************************
Sorting in ptseries file
************************
    Parcel 1:             CEREBELLUM_LEFT
                          8709 voxels
    Parcel 2:             THALAMUS_LEFT
                          1288 voxels
    Parcel 3:             CAUDATE_LEFT
                          728 voxels
    Parcel 4:             PUTAMEN_LEFT
                          1060 voxels
    Parcel 5:             PALLIDUM_LEFT
                          297 voxels
    Parcel 6:             BRAIN_STEM
                          3472 voxels
    Parcel 7:             HIPPOCAMPUS_LEFT
                          764 voxels
    Parcel 8:             AMYGDALA_LEFT
                          315 voxels
    Parcel 9:             ACCUMBENS_LEFT
                          135 voxels
    Parcel 10:            DIENCEPHALON_VENTRAL_LEFT
                          706 voxels
    Parcel 11:            CEREBELLUM_RIGHT
                          9144 voxels
    Parcel 12:            THALAMUS_RIGHT
                          1248 voxels
    Parcel 13:            CAUDATE_RIGHT
                          755 voxels
    Parcel 14:            PUTAMEN_RIGHT
                          1010 voxels
    Parcel 15:            PALLIDUM_RIGHT
                          260 voxels
    Parcel 16:            HIPPOCAMPUS_RIGHT
                          795 voxels
    Parcel 17:            AMYGDALA_RIGHT
                          332 voxels
    Parcel 18:            ACCUMBENS_RIGHT
                          140 voxels
    Parcel 19:            DIENCEPHALON_VENTRAL_RIGHT
                          712 voxels
%}
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

connectome_folder             = '/homeb/slns/slns019/HCP_MMP_Connectomes/FC/extracted_ptseries_txt/';
cd(connectome_folder)
subjects                      = load('/homec/hbu23/hbu231/HCP_Connectomes/SC/subject_list.txt');

SC_indices          = [361:365 379 366:378];
pt_indices          = [1:5 7:19 6];


for ii = sub_ii
    try       
        FC.id = subjects(ii);
        fMRI.id = subjects(ii);
        
        ts_MMP_REST1_LR          = load([num2str(subjects(ii)) '_FC_REST1_LR_MMP.ptseries.txt']);  
        ts_subco_REST1_LR        = load([num2str(subjects(ii)) '_FC_REST1_LR_subcort.ptseries.txt']); 
        ts_MMP_REST1_RL          = load([num2str(subjects(ii)) '_FC_REST1_RL_MMP.ptseries.txt']);  
        ts_subco_REST1_RL        = load([num2str(subjects(ii)) '_FC_REST1_RL_subcort.ptseries.txt']);  
        ts_MMP_REST2_LR          = load([num2str(subjects(ii)) '_FC_REST2_LR_MMP.ptseries.txt']);  
        ts_subco_REST2_LR        = load([num2str(subjects(ii)) '_FC_REST2_LR_subcort.ptseries.txt']); 
        ts_MMP_REST2_RL          = load([num2str(subjects(ii)) '_FC_REST2_RL_MMP.ptseries.txt']);  
        ts_subco_REST2_RL        = load([num2str(subjects(ii)) '_FC_REST2_RL_subcort.ptseries.txt']);   
        

        fMRI.MMP_REST1_LR       = [ts_MMP_REST1_LR; ts_subco_REST1_LR(pt_indices,:)]';
        fMRI.MMP_REST1_RL       = [ts_MMP_REST1_RL; ts_subco_REST1_RL(pt_indices,:)]';
        fMRI.MMP_REST2_LR       = [ts_MMP_REST2_LR; ts_subco_REST2_LR(pt_indices,:)]';
        fMRI.MMP_REST2_RL       = [ts_MMP_REST2_RL; ts_subco_REST2_RL(pt_indices,:)]';

        FC.MMP_REST1_LR        = corr(fMRI.MMP_REST1_LR);
        FC.MMP_REST1_RL        = corr(fMRI.MMP_REST1_RL);
        FC.MMP_REST2_LR        = corr(fMRI.MMP_REST2_LR);
        FC.MMP_REST2_RL        = corr(fMRI.MMP_REST2_RL);        
        
        FC.MMP_REST1_LR_fisherz    = atanh(FC.MMP_REST1_LR);
        FC.MMP_REST1_RL_fisherz    = atanh(FC.MMP_REST1_RL);
        FC.MMP_REST2_LR_fisherz    = atanh(FC.MMP_REST2_LR);
        FC.MMP_REST2_RL_fisherz    = atanh(FC.MMP_REST2_RL);      
        
        tmp = (FC.MMP_REST1_LR_fisherz + FC.MMP_REST1_RL_fisherz + FC.MMP_REST2_LR_fisherz + FC.MMP_REST2_RL_fisherz) / 4;
        n=size(tmp,1);
        tmp(1:n+1:n*n)=0;
        FC.MMP_fisherz_avg = tmp;

        
        tmp = (FC.MMP_REST1_LR + FC.MMP_REST1_RL + FC.MMP_REST2_LR + FC.MMP_REST2_RL) / 4;
        n=size(tmp,1);
        tmp(1:n+1:n*n)=0;
        FC.MMP_avg = tmp;
        
        save('-7',['/homeb/slns/slns019/HCP_MMP_Connectomes/FC/FC/' num2str(subjects(ii)) '_MMP_fMRI.mat'],'fMRI')
        save('-7',['/homeb/slns/slns019/HCP_MMP_Connectomes/FC/FC/' num2str(subjects(ii)) '_MMP_FC.mat'],'FC')

        
    catch me
        display([num2str(subjects(ii)) ': ' me.identifier])
    end
    

end



end


%%
% 
% cd('/Users/michael/Desktop/HCP_project/MMP_Connectomes/FC/scripts/')
% subjects            =   load('/Users/michael/Desktop/HCP_project/subject_list.txt');
% 
% initstring = ['#!/bin/bash -x\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --time=00:10:00\n#SBATCH --partition=batch\nmodule load GCC/7.2.0  ParaStationMPI/5.2.0-1\nmodule load Octave/4.2.1\n\n'];
% batch_line = 'srun -n 1 octave --eval "step2_extract_FC(';
% fileID_start = fopen(['step3_extractFC'],'w');
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
%     command      = [command batch_line num2str(ii)  ')" &\n'];
% 
%     if break_counter == 24 || ii == length(subjects)
%         preproc_patch = preproc_patch + 1;
%         finalstring = [command '\nwait\n'];
%         fileID = fopen(['step3_extractFC_' num2str(preproc_patch)],'w');
%         fprintf(fileID,finalstring);
%         fclose(fileID);
%         fprintf(fileID_start, ['sbatch step3_extractFC_' num2str(preproc_patch) '\n']);
%         break_counter = 0;
%     end
% end
% 
% fclose(fileID_start)
% 

