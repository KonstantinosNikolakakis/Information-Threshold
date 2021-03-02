% Author: Konstantinos Nikolakakis
% Paper Title: Optimal Rates for Learning Hidden Tree Structures
% Estimating the tree-structure of noisy and noiseless binary data
% Noise model: BSC(q), q = [0, 0.03, 0.06, 0.15]
% Output: Error-Rate (probability of incorrect strcture recovery)
 
clear all;
close all;

rng('shuffle')
Npoints=12; %Define the step difference for the correlation values, the correlations and the cross-over determine the value of information threshold

samples_batches=5;
%%%%%%%%% error-rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_zero_one_lossQ0=zeros(Npoints,samples_batches); 
acc_zero_one_lossQ1=zeros(Npoints,samples_batches);
acc_zero_one_lossQ2=zeros(Npoints,samples_batches);
acc_zero_one_lossQ3=zeros(Npoints,samples_batches);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

averaging=350; 
for accum=1:averaging

    %%% The tree is generated here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p=10; %number of vertices
    i=1:p;
    j=zeros(1,p);
    for k=2:p  %Generate a tree randomly
        j(k)=  ceil((k-1)*rand());
    end 
     adjacency=zeros(p);
    for k=2:p
        adjacency(i(k),j(k))=1;
    end    
    adjacency=adjacency'+adjacency;
    save('adjacency.mat','adjacency')
    
    %%%%%%%%% Visual representation of the tree %%%%%%%%%%%%%%%%%%%%%%%%%%
    %[ Tree,Cost1 ] =  UndirectedMaximumSpanningTree (adjacency );
    %bg1 = biograph(Tree);%bg1 = biograph(Tree,ids1);
    %get(bg1.nodes,'ID');
    %view(bg1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for points=1:Npoints
        %%% Determine the correlations and interaction parameters
        step=0.02*points;
        mu_s=0.99-step; % strong edges: large correlation namely mu_s
        mu_w=0.06+step; % weak edges: small correlation namely mu_w
        alpha=atanh(mu_w); % interaction parameter, weak edges
        beta=atanh(mu_s);  % interaction parameter, strong edges
        theta=zeros(p,p);  % initializing the matrix of interactions
        
        for k=1:p   %choose weak and strong edges with probability 1/2
            if j(k)~=0
                if rand()>0.5
                     theta(i(k),j(k))=beta;
                else
                    theta(i(k),j(k))=alpha;
                end
            end
        end
        theta=theta+theta';

        %%% Generate tree-structured samples %%%%%%%% 
        E=tanh(theta); %correlation matrix
        pairmarginals=zeros(size(E));
        pairmarginals(E~=0)=(1+E(E~=0))/2; %pairwise marginal p(X_i * X_j=+1)
        total_number_of_samples=1000000;
        
        samples=zeros(total_number_of_samples,p);

        for number_of_samples=1:total_number_of_samples
                    pairmarg=pairmarginals;
                    ii=i;
                    jj=j;
                    %Generate samples for the products X_i*X_j, these are
                    %independent random variables, for all edges (i,j)
                    Prodmatrix=zeros(p);
                    for r=2:p
                        if rand()<=pairmarg(ii(r),jj(r))
                            Prodmatrix(ii(r),jj(r))=1;
                        else
                            Prodmatrix(ii(r),jj(r))=-1;
                        end
                    end
                    Prodmatrix=Prodmatrix+Prodmatrix';
                    
                    %Generate a sample for the root independetly from the
                    %products X_i*X_j, since E[X_i' X_i X_j]=0
                    samplestemp=zeros(1,p);
                    if rand()>0.5
                       samplestemp(1)=1;
                    else
                       samplestemp(1)=-1;
                    end 
                    % Given the independent samples of the root and the sample of the prodacts
                    % we derive the samples of the rest of the vertices
                    for k=1:p-1
                        I = find(jj==k);
                        ell=length(I);
                        for m=1:ell
                            samplestemp(I(m))= samplestemp(k)*Prodmatrix(I(m),k);
                        end
                    end    

                    samples(number_of_samples,:)=samplestemp;
        end
        
        %%% Evaluating the noiseless information threshold%%%%%%%%%%%%%%%%%
        Ith(points)= 0.5* log2(((1-mu_w)^(1-mu_w)) * ((1+mu_w)^(1+mu_w))) - 0.5* log2(((1-mu_w*mu_s)^(1-mu_w*mu_s)) * ((1+mu_w*mu_s)^(1+mu_w*mu_s)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        runs=100;
        N=total_number_of_samples/runs;
        
        qcounter=0;
        q=0; %%% estimation from noiseless data
            qcounter=qcounter+1;
            sigma=(1-2*q)^2;
            IthNoisy0(qcounter)= 0.5* log2(((1-sigma*mu_w)^(1-sigma*mu_w)) * ((1+sigma*mu_w)^(1+sigma*mu_w))) - 0.5* log2(((1-sigma*mu_w*mu_s)^(1-sigma*mu_w*mu_s)) * ((1+sigma*mu_w*mu_s)^(1+sigma*mu_w*mu_s))); 
            noise=ones(total_number_of_samples,p);
            noise(q*ones(total_number_of_samples,p)>rand(total_number_of_samples,p))=-1;
            noisy=noise.*samples;
            
            ncounter=0;
            for n=2000:2000:N %estimate the structure for different batches
                ncounter=ncounter+1;
                temperror=zeros(1,runs);
                tempzero_one_loss=zeros(1,runs);
                for iter=1:runs
                    noisy_in=noisy;
                    Corr_matrix_estimate_noisy=noisy_in(n*(iter-1)+1:n*(iter),:)'*noisy_in(n*(iter-1)+1:n*(iter),:)/n;
                    x=Corr_matrix_estimate_noisy;
                    %%% Running the maximum spanning tree with input
                    %%% the estimated mututal information 
                    [Tree_est_noisy,Cost2] = UndirectedMaximumSpanningTree (0.5* log2(((1-x).^(1-x)) .* ((1+x).^(1+x))));
                   
                    Error=nnz(adjacency-Tree_est_noisy)/4;
                    temperror(iter)=Error/runs;
                    if Error~=0
                        tempzero_one_loss(iter)=1/runs;
                    end    
                end
                Error_q_n_matrixQ0(points,ncounter)=sum(temperror); % missed edges
                zero_one_lossQ0(points,ncounter)=sum(tempzero_one_loss); % error rate 
            end
        
        %%% similarly for the noisy cases
        q=0.03;
            qcounter=qcounter+1;
            sigma=(1-2*q)^2;
            %%%Find the theoretical value of infromation threshold
            IthNoisy1(points)= 0.5* log2(((1-sigma*mu_w)^(1-sigma*mu_w)) * ((1+sigma*mu_w)^(1+sigma*mu_w))) - 0.5* log2(((1-sigma*mu_w*mu_s)^(1-sigma*mu_w*mu_s)) * ((1+sigma*mu_w*mu_s)^(1+sigma*mu_w*mu_s))); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            noise=ones(total_number_of_samples,p);
            noise(q*ones(total_number_of_samples,p)>rand(total_number_of_samples,p))=-1; % flip samples with probability q
            noisy=noise.*samples;
            
            ncounter=0;
            for n=2000:2000:N
                ncounter=ncounter+1;
                temperror=zeros(1,runs);
                tempzero_one_loss=zeros(1,runs);
                for iter=1:runs
                    noisy_in=noisy;
                    Corr_matrix_estimate_noisy=noisy_in(n*(iter-1)+1:n*(iter),:)'*noisy_in(n*(iter-1)+1:n*(iter),:)/n;
                    x=Corr_matrix_estimate_noisy;
                    [Tree_est_noisy,Cost2] = UndirectedMaximumSpanningTree (0.5* log2(((1-x).^(1-x)) .* ((1+x).^(1+x))));
                   % bg2 = biograph(Tree_est_noisy);%bg1 = biograph(Tree,ids1);
                    %get(bg2.nodes,'ID');
                    %view(bg2);

                    Error=nnz(adjacency-Tree_est_noisy)/4;
                    temperror(iter)=Error/runs;
                    if Error~=0
                        tempzero_one_loss(iter)=1/runs;
                    end    
                end
                Error_q_n_matrixQ1(points,ncounter)=sum(temperror);
                zero_one_lossQ1(points,ncounter)=sum(tempzero_one_loss);
            end
            
            
        q=0.06;
            qcounter=qcounter+1;
            sigma=(1-2*q)^2;
            IthNoisy2(points)= 0.5* log2(((1-sigma*mu_w)^(1-sigma*mu_w)) * ((1+sigma*mu_w)^(1+sigma*mu_w))) - 0.5* log2(((1-sigma*mu_w*mu_s)^(1-sigma*mu_w*mu_s)) * ((1+sigma*mu_w*mu_s)^(1+sigma*mu_w*mu_s))); 
            noise=ones(total_number_of_samples,p);
            noise(q*ones(total_number_of_samples,p)>rand(total_number_of_samples,p))=-1;
            noisy=noise.*samples;
            ncounter=0;
            for n=2000:2000:N
                ncounter=ncounter+1;
                temperror=zeros(1,runs);
                tempzero_one_loss=zeros(1,runs);
                for iter=1:runs
                    noisy_in=noisy;
                    Corr_matrix_estimate_noisy=noisy_in(n*(iter-1)+1:n*(iter),:)'*noisy_in(n*(iter-1)+1:n*(iter),:)/n;
                    x=Corr_matrix_estimate_noisy;
                    [Tree_est_noisy,Cost2] = UndirectedMaximumSpanningTree (0.5* log2(((1-x).^(1-x)) .* ((1+x).^(1+x))));
                   % bg2 = biograph(Tree_est_noisy);%bg1 = biograph(Tree,ids1);
                    %get(bg2.nodes,'ID');
                    %view(bg2);

                    Error=nnz(adjacency-Tree_est_noisy)/4;
                    temperror(iter)=Error/runs;
                    if Error~=0
                        tempzero_one_loss(iter)=1/runs;
                    end    
                end
                Error_q_n_matrixQ2(points,ncounter)=sum(temperror);
                zero_one_lossQ2(points,ncounter)=sum(tempzero_one_loss);
            end

            q=0.15;
            qcounter=qcounter+1;
            sigma=(1-2*q)^2;
            IthNoisy3(points)= 0.5* log2(((1-sigma*mu_w)^(1-sigma*mu_w)) * ((1+sigma*mu_w)^(1+sigma*mu_w))) - 0.5* log2(((1-sigma*mu_w*mu_s)^(1-sigma*mu_w*mu_s)) * ((1+sigma*mu_w*mu_s)^(1+sigma*mu_w*mu_s))); 
            noise=ones(total_number_of_samples,p);
            noise(q*ones(total_number_of_samples,p)>rand(total_number_of_samples,p))=-1;
            noisy=noise.*samples;
            
            ncounter=0;
            for n=2000:2000:N
                ncounter=ncounter+1;
                temperror=zeros(1,runs);
                tempzero_one_loss=zeros(1,runs);
                for iter=1:runs
                    noisy_in=noisy;
                    Corr_matrix_estimate_noisy=noisy_in(n*(iter-1)+1:n*(iter),:)'*noisy_in(n*(iter-1)+1:n*(iter),:)/n;
                    x=Corr_matrix_estimate_noisy;
                    [Tree_est_noisy,Cost2] = UndirectedMaximumSpanningTree (0.5* log2(((1-x).^(1-x)) .* ((1+x).^(1+x))));
                   % bg2 = biograph(Tree_est_noisy);%bg1 = biograph(Tree,ids1);
                    %get(bg2.nodes,'ID');
                    %view(bg2);

                    Error=nnz(adjacency-Tree_est_noisy)/4;
                    temperror(iter)=Error/runs;
                    if Error~=0
                        tempzero_one_loss(iter)=1/runs;
                    end
                end
                Error_q_n_matrixQ3(points,ncounter)=sum(temperror);
                zero_one_lossQ3(points,ncounter)=sum(tempzero_one_loss);
            end
    end

    acc_zero_one_lossQ0=acc_zero_one_lossQ0+zero_one_lossQ0/averaging;
    acc_zero_one_lossQ1=acc_zero_one_lossQ1+zero_one_lossQ1/averaging;
    acc_zero_one_lossQ2=acc_zero_one_lossQ2+zero_one_lossQ2/averaging;
    acc_zero_one_lossQ3=acc_zero_one_lossQ3+zero_one_lossQ3/averaging;
end
figure(1)
for k=2:1:samples_batches
   plot(Ith(2:Npoints),acc_zero_one_lossQ0(2:Npoints,k),'LineWidth',8)
   set(gca,'FontSize',20)
   [wd, ht] = deal(15, 10);
   set(gcf, 'PaperPosition', [0 0 wd ht]);
   set(gcf, 'PaperSize', [wd ht]);
   set(gca, 'LooseInset', get(gca,'TightInset'));
   hold on
end
legend({'$n=4\times10^3$','$n=6\times10^3$','$n=8\times10^3$','$n=10\times10^3$'},'Interpreter','latex','FontSize',20)
xlabel('Noiseless Information Threshold $\mathrm{I^o}$','Interpreter','latex','FontSize',20) 
ylabel('Error Rate $\delta$','Interpreter','latex','FontSize',20) 
xlim([0 0.0068])

figure(2)
k=samples_batches;
plot(Ith(2:Npoints) , acc_zero_one_lossQ0(2:Npoints,k),'LineWidth',8)
set(gca,'FontSize',20)
[wd, ht] = deal(15, 10);
set(gcf, 'PaperPosition', [0 0 wd ht]);
set(gcf, 'PaperSize', [wd ht]);
set(gca, 'LooseInset', get(gca,'TightInset'));
hold on
plot(IthNoisy1(2:Npoints) , acc_zero_one_lossQ1(2:Npoints,k),'LineWidth',8)
hold on
plot(IthNoisy2(2:Npoints) , acc_zero_one_lossQ2(2:Npoints,k),'LineWidth',8)
hold on
plot(IthNoisy3(2:Npoints) ,acc_zero_one_lossQ3(2:Npoints,k),'LineWidth',8)
legend({'$q=0$', '$q=0.03$','$q=0.06$', '$q=0.15$'},'Interpreter','latex','FontSize',20)
xlabel('Noisy Information Threshold $\mathrm{I^o_{\dagger}}$','Interpreter','latex','FontSize',20) 
ylabel('Error Rate $\delta$','Interpreter','latex','FontSize',20)
xlim([0 0.0068])

save('acc_zero_one_lossQ0.mat','acc_zero_one_lossQ0')
save('acc_zero_one_lossQ1.mat','acc_zero_one_lossQ1')
save('acc_zero_one_lossQ2.mat','acc_zero_one_lossQ2')
save('acc_zero_one_lossQ3.mat','acc_zero_one_lossQ3')
