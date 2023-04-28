clear
close all
clc

% Read HCP FreeSurfer statistics (needs to be obtained from HCP)
T_fs = readtable('freesurfer_unrestricted_....csv');
T_fs_study = table(T_fs.Subject);
T_fs_study.FS_IntraCranial_Vol = T_fs.FS_IntraCranial_Vol;
T_fs_study.FS_BrainSeg_Vol = T_fs.FS_BrainSeg_Vol;

Afs = table2array(T_fs_study);

% take the cube-root as done in Smith et al. Nat Neuro
Afs(:,2:3) = nthroot(Afs(:,2:3), 3);
save('study_data_fs.mat','Afs');

%%
clear
close all
clc

% Read HCP unrestricted demographics data (needs to be obtained from HCP)
T = readtable('demographicsunrestricted_....csv');

%%%
% https://wiki.humanconnectome.org/display/PublicData/HCP-YA+Data+Dictionary-+Updated+for+the+1200+Subject+Release

T_study = table(T.Subject);

% Select all interesting measures
T_study.Flanker_AgeAdj = T.Flanker_AgeAdj; % The Flanker task measures both a participant?s attention and inhibitory control.
T_study.CardSort_AgeAdj = T.CardSort_AgeAdj; % DCCS is a measure of cognitive flexibility
T_study.PicSeq_AgeAdj = T.PicSeq_AgeAdj; % Picture Sequence Memory Test for the assessment of episodic memory
T_study.ListSort_AgeAdj = T.ListSort_AgeAdj ; %
T_study.ProcSpeed_AgeAdj = T.ProcSpeed_AgeAdj; % Pattern Comparison Processing Speed Test
T_study.PicVocab_AgeAdj = T.PicVocab_AgeAdj; % Picture Vocabulary Test
T_study.ReadEng_AgeAdj = T.ReadEng_AgeAdj; % Oral Reading Recognition Test

T_study.Flanker_Unadj = T.Flanker_Unadj; % The Flanker task measures both a participant?s attention and inhibitory control.
T_study.CardSort_Unadj = T.CardSort_Unadj; % DCCS is a measure of cognitive flexibility
T_study.PicSeq_Unadj = T.PicSeq_Unadj; % Picture Sequence Memory Test for the assessment of episodic memory
T_study.ListSort_Unadj = T.ListSort_Unadj ; %
T_study.ProcSpeed_Unadj = T.ProcSpeed_Unadj; % Pattern Comparison Processing Speed Test
T_study.PicVocab_Unadj = T.PicVocab_Unadj; % Picture Vocabulary Test
T_study.ReadEng_Unadj = T.ReadEng_Unadj; % Oral Reading Recognition Test

T_study.PSQI_Score = T.PSQI_Score; % Pittsburgh Sleep Quality Questionnaire
T_study.PMAT24_A_CR = T.PMAT24_A_CR; % Penn Progressive Matrices: Number of Correct Responses
T_study.PMAT24_A_SI = T.PMAT24_A_SI; % Penn Progressive Matrices: Total Skipped Items 
T_study.PMAT24_A_RTCR = T.PMAT24_A_RTCR; % Penn Progressive Matrices: Median Reaction Time for Correct Responses 
T_study.DDisc_AUC_200 = T.DDisc_AUC_200; % Delay Discounting: Area Under the Curve for Discounting of $200
T_study.DDisc_AUC_40K = T.DDisc_AUC_40K; % Delay Discounting: Area Under the Curve for Discounting of $40,000
T_study.VSPLOT_TC = T.VSPLOT_TC; % Variable Short Penn Line Orientation: Total Number Correct
%T_study.VSPLOT_CRTE = T.VSPLOT_CRTE ; % Variable Short Penn Line Orientation: Median Reaction Time Divided by Expected Number of Clicks for Correct
%T_study.VSPLOT_OFF = T.VSPLOT_OFF ; % Variable Short Penn Line Orientation: Total Positions Off for All Trials
T_study.SCPT_SEN = T.SCPT_SEN ; %
T_study.SCPT_SPEC = T.SCPT_SPEC ; %
T_study.IWRD_TOT = T.IWRD_TOT ; %
%T_study.IWRD_RTC = T.IWRD_RTC ; %
T_study.ER40_CR = T.ER40_CR ; %
%T_study.ER40_CRT = T.ER40_CRT ; %
T_study.AngAffect_Unadj = T.AngAffect_Unadj ; %
T_study.AngHostil_Unadj = T.AngHostil_Unadj ; %
T_study.AngAggr_Unadj = T.AngAggr_Unadj ; %
T_study.FearAffect_Unadj = T.FearAffect_Unadj ; %
T_study.FearSomat_Unadj = T.FearSomat_Unadj ; %
T_study.Sadness_Unadj = T.Sadness_Unadj ; %
T_study.LifeSatisf_Unadj = T.LifeSatisf_Unadj ; %
T_study.MeanPurp_Unadj = T.MeanPurp_Unadj ; %
T_study.PosAffect_Unadj = T.PosAffect_Unadj ; %
T_study.Friendship_Unadj = T.Friendship_Unadj ; %
T_study.Loneliness_Unadj = T.Loneliness_Unadj ; %
T_study.PercHostil_Unadj = T.PercHostil_Unadj ; %
T_study.PercReject_Unadj = T.PercReject_Unadj ; %
T_study.EmotSupp_Unadj = T.EmotSupp_Unadj ; %
T_study.InstruSupp_Unadj = T.InstruSupp_Unadj ; %
T_study.PercStress_Unadj = T.PercStress_Unadj ; %
T_study.SelfEff_Unadj = T.SelfEff_Unadj ; %
%T_study.Endurance_AgeAdj = T.Endurance_AgeAdj ; %
%T_study.GaitSpeed_Comp = T.GaitSpeed_Comp ; %
T_study.Dexterity_AgeAdj = T.Dexterity_AgeAdj ; %
%T_study.Strength_AgeAdj = T.Strength_AgeAdj ; %
T_study.NEOFAC_A = T.NEOFAC_A ; %
T_study.NEOFAC_O = T.NEOFAC_O ; %
T_study.NEOFAC_C = T.NEOFAC_C ; %
T_study.NEOFAC_N = T.NEOFAC_N ; %
T_study.NEOFAC_E = T.NEOFAC_E ; %
T_study.Noise_Comp = T.Noise_Comp ; %
T_study.Odor_AgeAdj = T.Odor_AgeAdj ; %
T_study.Taste_AgeAdj = T.Taste_AgeAdj ; %
T_study.WM_Task_2bk_Acc = T.WM_Task_2bk_Acc ; %
T_study.WM_Task_2bk_Median_RT = T.WM_Task_2bk_Median_RT ; %
T_study.WM_Task_0bk_Acc = T.WM_Task_0bk_Acc ; %
T_study.WM_Task_0bk_Median_RT = T.WM_Task_0bk_Median_RT ; %
%T_study.Mars_Log_Score = T.Mars_Log_Score;
T_study.Mars_Final = T.Mars_Final;


% Convert table to matrix
A = table2array(T_study);

% recon-all version --> needed as confounder
reconver=T.fMRI_3T_ReconVrs;
which_newer=strcmp('r227',reconver);
which_older=strcmp('r177',reconver);

% up to now confounds variable consists of recon-all vers. brain volume,
% intracranial volume
confounds = nan(size(A,1),4);
confounds(:,1) = A(:,1);
confounds(which_newer,2) = 1;
confounds(which_older,2) = 0;

% load fs-stats (intracranial volume, brain volume)
load('study_data_fs.mat')
indexing = find(A(:,1)==Afs(:,1));
confounds(:,3:4) = Afs(indexing,2:3);


% select only those subjects for which we have complete information
incompl=isnan(confounds(:,2)) + isnan(confounds(:,3))+ isnan(confounds(:,4));
incompl=incompl>0;
A = A(~incompl,:);
confounds = confounds(~incompl,:);


% From which subjects do we have SC and FC?
%clear
% FC
cd('../FC');
files = dir('*FC.mat');
for ii = 1:length(files)
    sub_id_FC(ii) = str2double(strtok(files(ii).name,'_'));    
end
% SC
cd('../SC');
files = dir('*SC.mat');
for ii = 1:length(files)
    sub_id_SC(ii) = str2double(strtok(files(ii).name,'_'));    
end

% which are in both?
sub_id = intersect(sub_id_FC, sub_id_SC);

%%% Select only those subjects for which we have SC and FC
[C,ia,ib] = intersect(A(:,1),sub_id);
A_study = A(ia,:);
confounds_study = confounds(ia,:);

% remove subjects with incomplete tests (nan)
complete_subs = find(sum(isnan(A_study)')==0);
A_study = A_study(complete_subs,:);
confounds_study = confounds_study(complete_subs,:);

A_final = A_study(:,2:end);
confounds_final = confounds_study(:,2:end);
cc=corr(A_final);
figure;imagesc(cc);colorbar
cd('..');
save('study_data.mat')
save('study_data_agg.mat','A_final','confounds_final','A_study')
%%
%%% Find robust set of connections by computing average SC
subjects = A_study(:,1);
mean_SC = zeros(379,379);
for ii = 1:length(subjects)
    SC = load(['../SC/' num2str(subjects(ii)) '_SC.mat']);
    SC = SC.SC.weights;
    mean_SC = mean_SC + SC;    
end
mean_SC = mean_SC / length(subjects);
save('mean_SC.mat','mean_SC');

%%% use only those connections that are robustly existent in all subjects
subjects = A_study(:,1);
num_conn_SC = zeros(379,379);
for ii = 1:length(subjects)
    SC = load(['../SC/' num2str(subjects(ii)) '_SC.mat']);
    SC = SC.SC.weights;
    %SC(SC < median(SC(:))) = 0;
    SC(SC~=0)  = 1;
    num_conn_SC = num_conn_SC + SC;    
end

% use only connections that were present in at least 99 % of all subjects
mask_SC = num_conn_SC >= 0.99*length(subjects);

save('num_conn_SC.mat','num_conn_SC', 'mask_SC');

%% Construct brain model input 
clear
close all
clc
load('study_data.mat');
load('mean_SC.mat');
load('num_conn_SC.mat');

subjects = A_study(:,1);


% get max for normalization
th_SC     = mean_SC;
th_SC(~mask_SC) = 0;
norm_fact = max(sqrt(th_SC(:)));


for ii = 1:length(subjects)
    FC = load(['../FC/' num2str(subjects(ii)) '_FC.mat']);
    FC = FC.FC.MMP_avg;
    SC = load(['../SC/' num2str(subjects(ii)) '_SC.mat']);
    SC_len = SC.SC.distances;
    SC = SC.SC.weights;
    SC = sqrt(SC);
    SC = SC ./ norm_fact;
    SC = SC ./ 10;
    
    SC(~mask_SC) = 0;
    SC_len(~mask_SC) = 0;
    
    mkdir(['../BNM_models/' num2str(subjects(ii))]);
    mkdir(['../BNM_models/' num2str(subjects(ii)) '/input/']);    
    mkdir(['../BNM_models/' num2str(subjects(ii)) '/output/']);    
    output_filestem = ['../BNM_models/' num2str(subjects(ii)) '/input/'];
    Generate_TVBii_input_FFI(num2str(subjects(ii)), FC, SC, SC_len, output_filestem)
end

%% Concatenate SC matrices of all coherently existing connections 
clear
close all
clc

load('study_data.mat');
load('mean_SC.mat');
load('num_conn_SC.mat');

subjects = A_study(:,1);

% SC connections that are present in all subjects
coherent_SC = find(num_conn_SC == length(subjects));

% get max for normalization
th_SC     = mean_SC;
th_SC(~mask_SC) = 0;
norm_fact = max(sqrt(th_SC(:)));

agg_SC = nan(length(subjects),length(coherent_SC));

for ii = 1:length(subjects)
    SC = load(['../SC/' num2str(subjects(ii)) '_SC.mat']);
    SC = SC.SC.weights;
    SC = sqrt(SC);
    SC = SC ./ norm_fact;
    SC = SC ./ 10;
    SC = SC(:);
    SC = SC(coherent_SC);
    agg_SC(ii,:) = SC;
end

save('agg_coherent_SC.mat','subjects','agg_SC')

%% Concatenate FC matrices
cd('../FC');
subjects = A_study(:,1);
Isubdiag = find(tril(ones(379),-1));

concat_FC = zeros(length(subjects), length(Isubdiag));

for ii = 1:length(subjects)
    fc = load([num2str(subjects(ii)) '_FC.mat']);
    concat_FC(ii,:) = fc.FC.MMP_avg(Isubdiag);
end
cd('..')
save('concat_FCs.mat','concat_FC')


