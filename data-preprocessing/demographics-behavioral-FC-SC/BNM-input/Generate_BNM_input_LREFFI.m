%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert structural and functional connectome into BNM-C readable input
%
%
% Input:
% SC weights, distances, FC
% 
% Output:
% - BNM-C input files (five files: weights, distances, reg_ids, FC reduced, FC full)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Generate_BNM_input_LREFFI(subject_ID, FC, SC, SC_len, output_filestem)

    % Write first header line: number of regions
    SCsize          =   length(SC);    
    dlmwrite([output_filestem subject_ID '_SC_strengths.txt'],SCsize);
    dlmwrite([output_filestem subject_ID '_SC_distances.txt'],SCsize);
    dlmwrite([output_filestem subject_ID '_SC_regionids.txt'],SCsize);
    dlmwrite([output_filestem subject_ID '_FC.txt'],SCsize);
    dlmwrite([output_filestem subject_ID '_FCfull.txt'],'');

    % Write second header line in dist-file: maximum distance
    %maxdist=1;
    maxdist=max(SC_len(:));
    dlmwrite([output_filestem subject_ID '_SC_distances.txt'],maxdist,'delimiter',' ','-append');
    
    % Write SC
    for ii = 1:length(SC)   
        % Format connectivity
        inpregs                = find(SC(ii,:)>0);
        inpcaps                = SC(ii,inpregs);
        inpFC                  = FC(ii,inpregs);
        %inpdists               = ones(1,length(inpregs));  % CAUTION: NO TIME-DELAYS!  
        inpdists               = SC_len(ii,inpregs);   
        inpregs                = inpregs-1; % to get C style numbering of region indices

        % Alternatly write line with region id and number of incoming
        % connections ...
        cap_line    =   [(ii-1) length(inpregs)];
        dist_line   =   [(ii-1) length(inpregs)];
        inp_line    =   [(ii-1) length(inpregs)];
        FC_line     =   [(ii-1) length(inpregs)];
        dlmwrite([output_filestem subject_ID '_SC_strengths.txt'],cap_line,   'delimiter',' ','-append');
        dlmwrite([output_filestem subject_ID '_SC_distances.txt'],dist_line,  'delimiter',' ','-append');
        dlmwrite([output_filestem subject_ID '_SC_regionids.txt'],inp_line,   'delimiter',' ','-append');
        dlmwrite([output_filestem subject_ID '_FC.txt'],          FC_line,   'delimiter',' ','-append');

        % ... and actual connectivity information
        dlmwrite([output_filestem subject_ID '_SC_strengths.txt'],inpcaps,    'delimiter',' ','-append','precision','%.8f');
        dlmwrite([output_filestem subject_ID '_SC_distances.txt'],inpdists,   'delimiter',' ','-append','precision','%.8f');
        dlmwrite([output_filestem subject_ID '_SC_regionids.txt'],inpregs,    'delimiter',' ','-append');
        dlmwrite([output_filestem subject_ID '_FC.txt'],          inpFC,      'delimiter',' ','-append');
        
        % write full FC line
        dlmwrite([output_filestem subject_ID '_FCfull.txt'],          FC(ii,:),      'delimiter',' ','-append');
    end            
    
end

