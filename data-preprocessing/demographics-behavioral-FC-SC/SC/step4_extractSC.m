function step4_extractSC(file_idx)

    connectome_folder            = '/homeb/slns/slns019/HCP_MMP_Connectomes/SC/SC/';
    output_connectome_folder     = '/homeb/slns/slns019/HCP_MMP_Connectomes/SC/SC_mat/';

    cd(connectome_folder)
    connectome_files      = dir('*_SC.csv');

    connectome_w          = load(connectome_files(file_idx).name);
    connectome_l          = load([connectome_files(file_idx).name(1:end-4) '_len.csv']);

    SC.id                 = connectome_files(file_idx).name(1:6);
    SC.weights            = zeros(size(connectome_w));
    SC.distances          = zeros(size(connectome_w));

    for i1 = 1:size(connectome_w,1)
        for i2  = i1+1:size(connectome_w,2)
            SC.weights(i1,i2) = connectome_w(i1,i2);
            SC.weights(i2,i1) = connectome_w(i1,i2);


            SC.distances(i1,i2) = connectome_l(i1,i2);
            SC.distances(i2,i1) = connectome_l(i1,i2);
        end
    end

    output_file = [output_connectome_folder connectome_files(file_idx).name(1:end-4) '.mat'];
    save('-7',output_file,'SC')


end
