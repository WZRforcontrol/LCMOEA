function Offspring = LearnableDE4(Problem, Population, V, k, a)
% The learnable differential evolution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [CR, F,proM,disM] = deal(1.0, 0.5,1,20);
    Lower = Problem.lower;
	Upper = Problem.upper;

    CV = sum(max(0,Population.cons),2); 
    %meanCV = mean(CV);

    [INDEX,DIS] = Association(Population,V,k);
    beta = 0.5;
    if a > beta
        for i = 1:Problem.N
            if CV(INDEX(1,i)) < CV(INDEX(2,i))
                winner(i) = Population(INDEX(1,i));
                losser(i) = Population(INDEX(2,i));
            elseif CV(INDEX(1,i)) == CV(INDEX(2,i)) && DIS(INDEX(1,i),i) < DIS(INDEX(2,i),i)
                winner(i) = Population(INDEX(1,i));
                losser(i) = Population(INDEX(2,i));            
            else
                winner(i) = Population(INDEX(2,i));            
                losser(i) = Population(INDEX(1,i));            
            end
        end 
    else
        for i = 1:Problem.N
            if DIS(INDEX(1,i),i) < DIS(INDEX(2,i),i)
                winner(i) = Population(INDEX(1,i));
                losser(i) = Population(INDEX(2,i));           
            else
                winner(i) = Population(INDEX(2,i));            
                losser(i) = Population(INDEX(1,i));            
            end
        end
    end
                
    
    mlp = ModelLearning(Problem, losser, winner);

    FrontNo = NDSort(Population.objs,Population.cons,1);   
    index1  = find(FrontNo==1);
    r       = floor(rand*length(index1))+1;
    best    = index1(r);

    %% Learnable Evolutionary Search for Reproduction
    % For each solution in the current population
    for i = 1 : Problem.N
        % Choose two different random parents
	    p = randperm(Problem.N, 3); 
		while p(1)==i || p(2)==i || p(3)==i
		    p = randperm(Problem.N, 3); 
        end	    	
        % Generate an child
        Parent1 = Population(i).decs;
		Parent2 = Population(p(1)).decs;
		Parent3 = Population(p(2)).decs;
        Parent4 = Population(p(3)).decs;
        Best_index = Population(best).decs;
        [~, D] = size(Parent1);
        child = Parent1;
        Site = rand(1,D) < CR; %CR = 1.0
        if CV(i) > 0.0001 && rand < 0.5
            [GDV, ~] = mlp.forward(child);
            GDV = GDV.*repmat(Upper-Lower,size(GDV,1),1) + repmat(Lower,size(GDV,1),1);
            if rand < 0.35
                child(Site) = child(Site) + F*(GDV(Site)-Parent1(Site)) + F*(Parent2(Site)-Parent3(Site));
            elseif rand < 0.7
                child(Site) = GDV(Site) + F*(Parent2(Site)-Parent3(Site));
            else
                child(Site) = child(Site) + F*(Best_index(Site)-Parent1(Site)) + F*(Parent2(Site)-Parent3(Site));
            end
        elseif CV(i) == 0
            if rand < 0.5
                [GDV, ~] = mlp.forward(child);
                GDV = GDV.*repmat(Upper-Lower,size(GDV,1),1) + repmat(Lower,size(GDV,1),1);
                child(Site) = GDV(Site) + F*(Parent2(Site)-Parent3(Site));
            else
                child(Site) = child(Site) + F*(Parent4(Site)-Parent1(Site)) + F*(Parent2(Site)-Parent3(Site));
            end
        else
            child(Site) = child(Site) + F*(Parent2(Site)-Parent3(Site));
        end                

        %% Polynomial mutation
        Site  = rand(1,D) < proM/D;
        mu    = rand(1,D);
        temp  = Site & mu<=0.5;
        child       = min(max(child,Lower),Upper);
        child(temp) = child(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
		            (1-(child(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp = Site & mu>0.5; 
        child(temp) = child(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
		            (1-(Upper(temp)-child(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
		
        %Evaluation of the new child
        child = Problem.Evaluation(child);
		
        %add the new child to the offspring population
        Offspring(i) = child;    
    end
    
end

function [INDEX,DIS] = Association(Population,V,k)
    % Normalization 
    N = length(Population);
    zmin = min(Population.objs,[],1);
    zmax = max(Population.objs,[],1);
    PopObj    = (Population.objs - repmat(zmin,N,1))./(repmat(zmax-zmin,N,1));
    % Associate k candidate solutions to each reference vector
    normP  = sqrt(sum(PopObj.^2,2));
    Cosine = 1 - pdist2(PopObj,V,'cosine');
    d1     = repmat(normP,1,size(V,1)).*Cosine;
    d2     = repmat(normP,1,size(V,1)).*sqrt(1-Cosine.^2);
    DIS    = d1 + 3.0*d2;
    [~,index] = sort(d2,1);
    INDEX     = index(1:min(k,length(index)),:);
end