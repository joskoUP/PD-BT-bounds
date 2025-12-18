% This is a script to create plots for PD-(TL)BT for the ISS1R model and
% the related posterior error bounds. The code for the benchmark example
% and PD-BT is taken from https://github.com/joskoUP/PD-BT. 
% (Requires lyapchol from control system toolbox).

clear; close all
rng(1)
%% define LTI model
% ISS model
load('iss1R.mat')
d           = size(A,1);
d_out       = size(C,1);
sig_obs     = [2.5e-3 5e-4 5e-4]';

% define time frame (measurement times) for the inference problem
T               = 8; % end time
dt_obs          = 0.1; % difference between measurements   
n               = round(T/dt_obs); % number of measurements
sig_obs_long    = repmat(sig_obs,n,1);

%% given low-rank prior - sample covariance of compatible covariance
ensemble_size   = 90; % size of prior ensemble
Lyap_solution   = lyapchol(A,B)'; % compatibility Lyapunov equation
ensemble        = Lyap_solution * randn(d,ensemble_size); 
% ensemble drawn from compatible covariance 
Gamma_pr        = cov(ensemble'); % prior is ensemble covariance
prior_rank      = rank(Gamma_pr); % rank of prior covariance
% compute a low-rank factor of the prior covariance
[X,D]           = eig(Gamma_pr); % eigendecompostition of prior
[~,ind]         = sort(diag(D),'descend'); % get important directions
Ds              = D(ind,ind);
Xs              = X(:,ind);
Gamma_pr_root   = Xs(:,1:prior_rank)*sqrt(Ds(1:prior_rank,1:prior_rank));

figure; clf
%% set up experiments with differently scaled priors
scaling_factors = [0.1,1,10];
for scale_count = 1:length(scaling_factors)
    L_pr    = scaling_factors(scale_count)*Gamma_pr_root;
    % prior for experiment is scaled version of Gamma_pr

    %% compute infinite noisy observability Gramian
    % helper matrix    
    F           = C./sig_obs;
    L_Q_inf     = lyapchol(A',F')';

    %% compute infinite prior-driven reachability Gramian
    L_P_inf     = lyapchol(A,L_pr)';  

    %% compute time-limited noisy observability Gramian
    right_sideQ = -F'*F+expm(A'*T)*(F'*F)*expm(A*T);
    Q           = sylvester(A',A,right_sideQ);
    L_Q         = real(sqrtm(Q)); 
    
    %% compute time-limited prior-driven reachability Gramian
    right_side  = -(L_pr*L_pr')+expm(A*T)*(L_pr*L_pr')*expm(A'*T);
    P           = sylvester(A,A',right_side);
    L_P         = real(sqrtm(P));
    
    %% set up the experiment
    % generate random initial condition
    x0          = L_pr*randn(size(L_pr,2),1);
    
    % define full forward model 
    G       = zeros(n*d_out,d);
    iter    = expm(A*dt_obs);
    temp    = C;
    for i   = 1:n
        temp                        = temp*iter;
        G((i-1)*d_out+1:i*d_out,:)  = temp;
    end
    % compute Fisher info
    Go  = G./sig_obs_long;
    H   = Go'*Go;
    
    % helper matrix
    M   = G*L_pr;
    
    % generate measurements and add noise
    y   = G*x0;
    m   = y + sig_obs_long.*randn(n*d_out,1);
    m_scaled = m./sig_obs_long;
    
    %% compute true posterior
    full_rhs        = Go'*m_scaled;
    PosCov_true     = L_pr*(eye(size(L_pr,2)) - M'*((M*M'+diag(sig_obs_long.^2))\M))*L_pr';
    PosMean_true    = PosCov_true*full_rhs;
    
    %% compute posterior approximations and errors
    r_vals      = 1:2:(prior_rank/2);
    rmax        = max(r_vals);
    
    % prior-driven balancing (PD-BT)
    [V,S,W]     = svd(L_Q_inf'*L_P_inf); 
    % compute balanced quantities for posterior computation and error bound
    delQ_inf    = diag(S);
    Siginvsqrt  = diag(1./sqrt(delQ_inf));
    Sr_inf      = (Siginvsqrt*V'*L_Q_inf')';
    Tr_inf      = L_P_inf*W*Siginvsqrt; % balancing transformation
    A_inf       = Sr_inf'*A*Tr_inf;
    C_inf       = C*Tr_inf;
    L_pr_inf    = Sr_inf'*L_pr;
    
    % time-limited prior-driven balancing (PD-TLBT)
    [V,S,W]     = svd(L_Q*L_P); 
    % compute balanced quantities for posterior computation and error bound
    delQ        = diag(S);
    Siginvsqrt  = diag(1./sqrt(delQ));
    Sr_TL       = (Siginvsqrt*V'*L_Q)';
    Tr_TL       = L_P*W*Siginvsqrt; % balancing transformation
    A_TL        = Sr_TL'*A*Tr_TL;
    C_TL        = C*Tr_TL;
    L_pr_TL     = Sr_TL'*L_pr;
    
    %% compute posterior approximations
    F_Dist_TL       = zeros(1,length(r_vals)); % PD-TLBT Frobenius error (full space)
    F_Dist_inf      = zeros(1,length(r_vals)); % PD-BT Frobenius error (full space)
    mean_Dist_TL    = zeros(1,length(r_vals)); % PD-TLBT error on first mean (full Space)
    mean_Dist_inf   = zeros(1,length(r_vals)); % PD-BT error on first mean (full Space)
    Trace_vals_TL   = zeros(1,length(r_vals)); % trace for PD-TLBT error bound
    Trace_vals_inf  = zeros(1,length(r_vals)); % trace for PD-BT error bound
    
    for rr = 1:length(r_vals)
        r = r_vals(rr);
        %% PD-BT posterior quantities
        % PD-BT - generate reduced forward matrix G_inf
        G_inf               = zeros(n*d_out,r);
        iter                = expm(A_inf(1:r,1:r)*dt_obs);
        temp                = C_inf(:,1:r);
        for i = 1:n
            temp                             = temp*iter;
            G_inf((i-1)*d_out+1:i*d_out,:)   = temp;
        end
        G_infS              = G_inf*Sr_inf(:,1:r)';
        M_inf               = G_inf*L_pr_inf(1:r,:); % reduced helper matrix
        
        % PD-BT - compute posterior covariance
        PosCov_inf          = L_pr*(eye(size(L_pr,2)) - M_inf'*((M_inf*M_inf'+diag(sig_obs_long.^2))\M_inf))*L_pr';
        F_Dist_inf(rr)      = norm(PosCov_inf-PosCov_true,'fro');
        % PD-BT - compute posterior mean
        mean_inf            = PosCov_inf*(G_infS./sig_obs_long)'*m_scaled;
        mean_Dist_inf(rr)   = norm(mean_inf-PosMean_true);
        
        %% PD-BT - assemble necessary quantities for the outpur error bound
        L2_inf              = L_pr_inf(r+1:d,:);
        C1_inf              = C_inf(:,1:r)./sig_obs;
        A11_inf             = A_inf(1:r,1:r);
        A12_inf             = A_inf(1:r,r+1:d);
        Y_inf               = sylvester(A_inf',A11_inf,-C_inf'*C1_inf);
        Y2_inf              = Y_inf(r+1:d,:);
        delQ2_inf           = diag(delQ_inf(r+1:d));
        Trace_vals_inf(rr)  = trace((L2_inf*L2_inf'+2*Y2_inf*A12_inf)*delQ2_inf);
    
        %% time-limited PD-BT posterior quantities
        % PD-BT - generate reduced forward matrix G_TL
        G_TL            = zeros(n*d_out,r);
        iter            = expm(A_TL(1:r,1:r)*dt_obs);
        temp            = C_TL(:,1:r);
        for i = 1:n
            temp                             = temp*iter;
            G_TL((i-1)*d_out+1:i*d_out,:)   = temp;
        end
        G_TLS           = G_TL*Sr_TL(:,1:r)';
        M_TL            = G_TL*L_pr_TL(1:r,:); % reduced helper matrix
        
        % PD-BT - compute posterior covariance
        PosCov_TL       = L_pr*(eye(size(L_pr,2)) - M_TL'*((M_TL*M_TL'+diag(sig_obs_long.^2))\M_TL))*L_pr';
        F_Dist_TL(rr)   = norm(PosCov_TL-PosCov_true,'fro');
        % PD-BT - compute posterior mean
        mean_TL         = PosCov_TL*(G_TLS./sig_obs_long)'*m_scaled;
        mean_Dist_TL(rr)= norm(mean_TL(:,1)-PosMean_true(:,1));
        
        %% time-limited PD-BT - assemble necessary quantities for the outpur error bound
        L1      = L_pr_TL(1:r,:);
        A11     = A_TL(1:r,1:r);
        C1      = C_TL(:,1:r)./sig_obs;
        % compute mixed Gramian
        right_side_M    = -L_pr*L1'+expm(A*T)*(L_pr*L1')*expm(A11'*T);
        PTM             = real(sylvester(A,A11',right_side_M)); 
        % compute reduced Gramian
        right_side_r        = -L1*L1'+expm(A11*T)*(L1*L1')*expm(A11'*T);
        PTr                 = real(sylvester(A11,A11',right_side_r)); 
        Trace_vals_TL(rr)   = trace(F*P*F')+trace(C1*PTr*C1')-2*trace(F*PTM*C1');
    end
    
    %% plots
    % plot PD-BT posterior covariance error bound    
    subplot(3,2,2*scale_count)
    semilogy(r_vals,norm(Gamma_pr_root)^2*sqrt(Trace_vals_TL),'--','Color','#DC267F','LineWidth',3); hold on
    semilogy(r_vals,norm(Gamma_pr_root)^2*sqrt(Trace_vals_inf),'--','Color','#648FFF','LineWidth',2);
    semilogy(r_vals,F_Dist_TL,'-','Color','#DC267F','LineWidth',3)
    semilogy(r_vals,F_Dist_inf,'-','Color','#648FFF','LineWidth',2)
    xlim([0 rmax])
    ylim([1e-6 1e+5])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')
    if scale_count==1
        title('Posterior covariance error and bound','interpreter','latex','fontsize',20)
    end
    if scale_count==length(scaling_factors)
        xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'error bound TLBT','error bound PD-BT','$\|\mathbf{\Gamma}_\mathrm{pos}-\hat{\mathbf{\Gamma}}_\mathrm{pos}\|_F$ PD-TLBT','$\|\mathbf{\Gamma}_\mathrm{pos}-\hat{\mathbf{\Gamma}}_\mathrm{pos}\|_F$ PD-BT'},'interpreter','latex','fontsize',13,'Location','best')
        legend boxoff
    end

    % plot PD-BT posterior mean error bound    
    subplot(3,2,2*scale_count-1)
    semilogy(r_vals,norm(Gamma_pr_root)^2*sqrt(Trace_vals_TL),'--','Color','#DC267F','LineWidth',3); hold on
    semilogy(r_vals,norm(Gamma_pr_root)^2*sqrt(Trace_vals_inf),'--','Color','#648FFF','LineWidth',2);
    semilogy(r_vals,mean_Dist_TL,'-','Color','#DC267F','LineWidth',3);
    semilogy(r_vals,mean_Dist_inf,'-','Color','#648FFF','LineWidth',2);
    xlim([0 rmax])
    ylim([1e-6 1e+5])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')
    ylabel(['$\mathbf{\Gamma}_\mathrm{pr}=$',num2str(scaling_factors(scale_count)^2),'$\cdot\mathbf{\Gamma}$'],'interpreter','latex','fontsize',13)
    if scale_count==1
        title('Posterior mean error and bound','interpreter','latex','fontsize',20)
    end
    if scale_count==length(scaling_factors)
        xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'error bound TLBT','error bound PD-BT','$\|\mathbf{\mu}_\mathrm{pos}-\hat{\mathbf{\mu}}_\mathrm{pos}\|_2$ PD-TLBT','$\|\mathbf{\mu}_\mathrm{pos}-\hat{\mathbf{\mu}}_\mathrm{pos}\|_2$ PD-BT'},'interpreter','latex','fontsize',13,'Location','best')
        legend boxoff
    end
end