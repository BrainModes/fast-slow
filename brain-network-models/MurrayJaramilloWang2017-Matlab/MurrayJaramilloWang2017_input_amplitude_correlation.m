% MurrayJaramilloWang JNeuro 2017 model equations
function return_table=MurrayJaramilloWang2017_input_amplitude_correlation(corr_val_inp)
    
    rng default % Reset random number generator
    %rng(seed)
    
    % Simulation parameters
    
    dt      = 1;
    ts      = 3000; % (ms) simulation time
    tss     = length(1:dt:ts);
    
    
    
    % Model parameters
    tau             = 60;       % (ms) NMDA time constant
    gamma           = 0.641;    % rate of saturation of S
    I_0             = 0.334;    % (nA) background current
    tau_AMPA        = 2;        % (ms) AMPA noise time constant
    sigma_noise     = 0.009;    % (nA) noise strength
    
    
    % Connection weights
    % Population sorting A_1, B_1, A_2, B_2
    % Local Module 1: 
    % J_same = (J_S + J_T) / 2 = (0.35 + 0.28387) / 2           = 0.3169
    % J_diff = J_T - J_same = 0.28387 - 0.3169                  = -0.0330
    % Local Module 2: 
    % J_same = (J_S + J_T) / 2 = (0.4182 + 0.28387) / 2         = 0.351
    % J_diff = J_T - J_same = 0.28387 - 0.351                   = -0.0671
    % Long-range Module 1 -> Module 2:
    % J_same = (J_S + J_T) / 2 = (0.15 + 0) / 2                 = 0.075
    % J_diff = J_T - J_same = 0 - 0.0750                        = -0.075
    % Long-range Module 1 -> Module 1:
    % J_same = (J_S + J_T) / 2 = (0.04 + 0) / 2                 = 0.02
    % J_diff = J_T - J_same = 0 - 0.0200                        = -0.02
    
    
    N  = 5000;
    
    % output variables
    S_outA1   = zeros(ts,N);
    S_outB1   = zeros(ts,N);
    S_outA2   = zeros(ts,N);
    S_outB2   = zeros(ts,N);
    
    J_same_M1 = 0.3169;
    J_diff_M1 = -0.0330;
    J_same_M2 = 0.351;
    J_diff_M2 = -0.0671;
    
    J_same_M1_to_M2 = 0.075;
    J_diff_M1_to_M2 = -0.075;
    J_same_M2_to_M1 = 0.02;
    J_diff_M2_to_M1 = -0.02;
    
    % Model variables
    S_A1     = zeros(N,1);
    S_B1     = zeros(N,1);
    S_A2     = zeros(N,1);
    S_B2     = zeros(N,1);
    I_noiseCC= zeros(N,1);
    I_noiseA = zeros(N,1);
    I_noiseB = zeros(N,1);
    I_noiseA2= zeros(N,1);
    I_noiseB2= zeros(N,1);
    
    
    %current_add = repmat(current_add,N_noise,1);
    contr = 0.2;
    %corr_val = 0:0.1:1;
    %corr_val = repmat(corr_val',1000,1);
    corr_val = corr_val_inp;
    
    for ii = 1:tss    
        
        if ii > 500 && ii < 2000
            %I_appA          = linspace(0.0, 0.07, N)';
            I_appA           = 0.0118 * (1 + contr / 100);
            I_appB           = 0.0118 * (1 - contr / 100);
        else
            I_appA          = 0.0;
            I_appB          = 0.0;
        end
        
        I_noiseCC =  I_noiseCC +  dt  *  (  (-I_noiseCC +  randn(N,1)  *  sqrt(tau_AMPA  *  sigma_noise.^2))  ./ tau_AMPA ); 
        I_noiseA  =  I_noiseA  +  dt  *  (  (-I_noiseA  +  randn(N,1)  *  sqrt(tau_AMPA  *  sigma_noise.^2))  ./ tau_AMPA ); 
        I_noiseB  =  I_noiseB  +  dt  *  (  (-I_noiseB  +  randn(N,1)  *  sqrt(tau_AMPA  *  sigma_noise.^2))  ./ tau_AMPA ); 
        I_noiseA2 =  I_noiseA2 +  dt  *  (  (-I_noiseA2 +  randn(N,1)  *  sqrt(tau_AMPA  *  sigma_noise.^2))  ./ tau_AMPA ); 
        I_noiseB2 =  I_noiseB2 +  dt  *  (  (-I_noiseB2 +  randn(N,1)  *  sqrt(tau_AMPA  *  sigma_noise.^2))  ./ tau_AMPA ); 
    
        % Generate correlated input noise
        I_noiseAf = corr_val .* I_noiseCC + (1 - corr_val) .* I_noiseA;
        I_noiseBf = corr_val .* I_noiseCC + (1 - corr_val) .* I_noiseB;    
        I_noiseAf2= corr_val .* I_noiseCC + (1 - corr_val) .* I_noiseA2;
        I_noiseBf2= corr_val .* I_noiseCC + (1 - corr_val) .* I_noiseB2;
        
        % set input amplitude offsets
        curr_addPPC = +0.00;
        curr_addPFC = -0.00;
    
        % input currents
        I_A1      =  J_same_M1       .* S_A1 + J_diff_M1       .* S_B1 + J_same_M2_to_M1 .* S_A2 + J_diff_M2_to_M1 .* S_B2 +  I_0  +  I_noiseAf  +  I_appA + curr_addPPC;
        I_B1      =  J_diff_M1       .* S_A1 + J_same_M1       .* S_B1 + J_diff_M2_to_M1 .* S_A2 + J_same_M2_to_M1 .* S_B2 +  I_0  +  I_noiseBf  +  I_appB + curr_addPPC;
        I_A2      =  J_same_M1_to_M2 .* S_A1 + J_diff_M1_to_M2 .* S_B1 + J_same_M2       .* S_A2 + J_diff_M2       .* S_B2 +  I_0  +  I_noiseAf2           + curr_addPFC;
        I_B2      =  J_diff_M1_to_M2 .* S_A1 + J_same_M1_to_M2 .* S_B1 + J_diff_M2       .* S_A2 + J_same_M2       .* S_B2 +  I_0  +  I_noiseBf2           + curr_addPFC;
    
        % synaptic activity
        S_A1      =  S_A1  +   dt * ( -S_A1 / tau  +  gamma * (1  -  S_A1) .* r(I_A1) / 1000);
        S_B1      =  S_B1  +   dt * ( -S_B1 / tau  +  gamma * (1  -  S_B1) .* r(I_B1) / 1000);
        S_A2      =  S_A2  +   dt * ( -S_A2 / tau  +  gamma * (1  -  S_A2) .* r(I_A2) / 1000);
        S_B2      =  S_B2  +   dt * ( -S_B2 / tau  +  gamma * (1  -  S_B2) .* r(I_B2) / 1000);
    
        %S(S<0)  = 0;    
        %S(S>1)  = 1;
        
        S_outA1(ii,:)  = r(I_A1);
        S_outB1(ii,:)  = r(I_B1);
        S_outA2(ii,:)  = r(I_A2);
        S_outB2(ii,:)  = r(I_B2);
    end
    
    
    correct_response   = mean(S_outA1(end-200:end,:)) > 20 & mean(S_outB1(end-200:end,:)) < 20;
    %correct_response_r = reshape(correct_response,11,1000)';
    
    wrong_response = mean(S_outB1(end-200:end,:)) > 20 & mean(S_outA1(end-200:end,:)) < 20;
    %wrong_response_r = reshape(wrong_response,11,1000)';
    
    return_table = mean(correct_response);

end

% input currents to firing rates
function phi = r(I)
    a       = 270;     % (Hz/nA)
    b       = 108;     % (Hz)
    c       = 0.154;   % (s)

    phi = (a .* I - b) ./ (1 - exp(-c .* (a .* I - b)));
end

