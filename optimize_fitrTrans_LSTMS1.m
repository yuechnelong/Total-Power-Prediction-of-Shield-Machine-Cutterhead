%% 
function [Mdl2, fMin, Convergence_curve, Loss2] = optimize_fitrTrans_LSTMS1(p_train1, train_y_feature_label_norm, p_vaild1, vaild_y_feature_label_norm, num_pop, num_iter, method_mti, max_epoch, min_batchsize, lstms_label)
   
    
    pop = num_pop;
    M = num_iter;
    LB = [0, 0, 2, 0.001, 0.001];  
    UB = [6, 6, 8, 0.5, 0.01];     
   
    nvars = length(LB);
    fit_fitrensemble1 = @fit_fitrTrans_LSTM; 
    

    if strcmp(method_mti, 'GRIME') == 1
        [fMin, Mdl, Convergence_curve, pos] = GRIME_GRIME(pop, M, LB, UB, nvars, fit_fitrensemble1, ...
            p_train1, train_y_feature_label_norm, p_vaild1, vaild_y_feature_label_norm, ...
            max_epoch, min_batchsize, lstms_label);
    end
    
    Mdl2 = Mdl.Mdl;
    Loss2 = Mdl.Loss;
    
  
   figure
    plot(Convergence_curve, '--p', 'LineWidth', 1.2, 'Color', [160, 123, 194] ./ 255)
    xticks(1:length(Convergence_curve))
    title('Optimization Process')
    xlabel('Iteration Count')
    ylabel('Fitness Value')
    grid on
    set(gca, "FontName", "Times New Roman", "FontSize", 12, "LineWidth", 1.2)
    box off
    
    if lstms_label == 3
        disp([method_mti, ' optimized Transformer-BiLSTM:   ', ...
            "Position Encoding Vector Count:", num2str(2^(round(pos(1)))), ...
            "   Attention Mechanism Heads:", num2str(2^(round(pos(2)))), ...
            "   Attention Mechanism Key Size:", num2str(2^(round(pos(2)) + 1)), ...
            "   Hidden Layer Neurons (BiLSTM):", num2str(2^(round(pos(3)))), ...
            '   Dropout Layer Rate: ', num2str((pos(4))), ...
            '   Learning Rate: ', num2str((pos(5)))])
    end
end

%% Transformer-BiLSTM
function [fitness_value, Mdl1] = fit_fitrTrans_LSTM(pop, p_train1, train_y_feature_label_norm, p_vaild1, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label)
  
 
    if lstms_label == 3
     
        layers = [
            sequenceInputLayer(length(p_train1{1, 1}), Name = "input")
            positionEmbeddingLayer(length(p_train1{1, 1}), 2^(round(pop(1))), Name = "pos-emb")
            additionLayer(2, Name = "add")
            selfAttentionLayer(2^(round(pop(2))), 2^(round(pop(2)) + 1))
            dropoutLayer(pop(4))                                
            selfAttentionLayer(2^(round(pop(2))), 2^(round(pop(2)) + 1))
            dropoutLayer(pop(4));
            bilstmLayer(2^(round(pop(3))), 'OutputMode', 'sequence') 
            fullyConnectedLayer(length(train_y_feature_label_norm{1, 1}))
            regressionLayer];

        lgraph = layerGraph(layers);
        layers = connectLayers(lgraph, "input", "add/in2");
    end
    
    options = trainingOptions('adam', ...
        'MaxEpochs', max_epoch, ...
        'MiniBatchSize', min_batchsize, ...
        'InitialLearnRate', pop(5), ...
        'ValidationFrequency', 20);
    
  
    [Mdl, Loss] = trainNetwork(p_train1, train_y_feature_label_norm, layers, options);
 
    P_vaild_y_feature_label_norm = predict(Mdl, p_vaild1, 'MiniBatchSize', min_batchsize);

    P_vaild_y_feature_label_norm1 = [];
    vaild_y_feature_label_norm1 = [];
    for i = 1:length(P_vaild_y_feature_label_norm)
        P_vaild_y_feature_label_norm1(i, :) = (P_vaild_y_feature_label_norm{i, 1});
        vaild_y_feature_label_norm1(i, :) = (vaild_y_feature_label_norm{i, 1});
    end
    
    fitness_value = sum(sum(abs(P_vaild_y_feature_label_norm1 - vaild_y_feature_label_norm1))) / length(vaild_y_feature_label_norm1);
 
    Mdl1.Mdl = Mdl;
    Mdl1.Loss = Loss;
end

%% GRIME
function [Best_rime_rate, Mdl_best, Convergence_curve, Best_rime] = GRIME_GRIME(N, Max_iter, lb, ub, dim, fobj, train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label)
   
    Best_rime = zeros(1, dim);
    Best_rime_rate = inf; 
    

    label = 5; 
    Rimepop = yinshe(N, dim, label, ub, lb);
    Rimepop = min(ub, Rimepop);
    Rimepop = max(lb, Rimepop);
    
    Lb = lb .* ones(1, dim); 
    Ub = ub .* ones(1, dim); 
    it = 1; %
    Convergence_curve = zeros(1, Max_iter);
    Rime_rates = zeros(1, N);
    newRime_rates = zeros(1, N);
    W = 5; 
    
   
    for i = 1:N
        [Rime_rates(1, i), Mdl_all{1, i}] = fobj(Rimepop(i, :), train_x_feature_label_norm, train_y_feature_label_norm, ...
            vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label);
       
        if Rime_rates(1, i) < Best_rime_rate
            Best_rime_rate = Rime_rates(1, i);
            Best_rime = Rimepop(i, :);
            Mdl_best = Mdl_all{1, i};
        end
    end
    
    
    while it <= Max_iter
        
        RimeFactor = (rand - 0.5) * 2 * cos((pi * it / (Max_iter / 10))) * (1 - round(it * W / Max_iter) / W);
        E = sqrt(it / Max_iter); 
        newRimepop = Rimepop;
        normalized_rime_rates = normr(Rime_rates); 
        
     
        for i = 1:N
            for j = 1:dim
                
                r1 = rand();
                if r1 < E
                    newRimepop(i, j) = Best_rime(1, j) + RimeFactor * ((Ub(j) - Lb(j)) * rand + Lb(j));
                end
                
               
                r2 = rand();
                if r2 < normalized_rime_rates(i)
                    newRimepop(i, j) = Best_rime(1, j);
                end
            end
        end
        
      
        for i = 1:N
            Flag4ub = newRimepop(i, :) > ub;
            Flag4lb = newRimepop(i, :) < lb;
            newRimepop(i, :) = (newRimepop(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            
            [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), train_x_feature_label_norm, train_y_feature_label_norm, ...
                vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label);
            
            
            if newRime_rates(1, i) < Rime_rates(1, i)
                Rime_rates(1, i) = newRime_rates(1, i);
                Rimepop(i, :) = newRimepop(i, :);
                Mdl_all{1, i} = Mdl_new{1, i};
                
                if newRime_rates(1, i) < Best_rime_rate
                    Best_rime_rate = Rime_rates(1, i);
                    Best_rime = Rimepop(i, :);
                    Mdl_best = Mdl_all{1, i};
                end
            end
        end
        
        
        for i = 1:N
            k = (1 + (it / Max_iter)^0.5)^10;
            newRimepop(i, :) = (ub + lb) / 2 + (ub + lb) / (2 * k) - Rimepop(i, :) / k;
            
            Flag4ub = newRimepop(i, :) > ub;
            Flag4lb = newRimepop(i, :) < lb;
            newRimepop(i, :) = (newRimepop(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            
            [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), train_x_feature_label_norm, train_y_feature_label_norm, ...
                vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label);
            
            if newRime_rates(1, i) < Rime_rates(1, i)
                Rime_rates(1, i) = newRime_rates(1, i);
                Rimepop(i, :) = newRimepop(i, :);
                Mdl_all{1, i} = Mdl_new{1, i};
                
                if newRime_rates(1, i) < Best_rime_rate
                    Best_rime_rate = Rime_rates(1, i);
                    Best_rime = Rimepop(i, :);
                    Mdl_best = Mdl_all{1, i};
                end
            end
        end
        
        Convergence_curve(it) = Best_rime_rate;
        it = it + 1;
    end
end

%% 
function result = yinshe(N, dim, label, ub, lb)
  
    
    if label == 1
        % tent
        tent = 1.2; 
        Tent = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                if Tent(i, j - 1) < tent
                    Tent(i, j) = Tent(i, j - 1) / tent;
                elseif Tent(i, j - 1) >= tent
                    Tent(i, j) = (1 - Tent(i, j - 1)) / (1 - tent);
                end
            end
        end
        result = lb + Tent .* (ub - lb);
        
    elseif label == 2
        % chebyshev
        chebyshev = 2;
        Chebyshev = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Chebyshev(i, j) = cos(chebyshev .* acos(Chebyshev(i, j - 1)));
            end
        end
        result = lb + (Chebyshev + 1) / 2 .* (ub - lb);
        
    elseif label == 3
        % singer
        u = 1;
        singer = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                singer(i, j) = u * (7.86 * singer(i, j - 1) - 23.31 * singer(i, j - 1).^2 + ...
                    28.75 * singer(i, j - 1).^3 - 13.302875 * singer(i, j - 1).^4);
            end
        end
        result = lb + singer .* (ub - lb);
        
    elseif label == 4
        % Logistic
        miu = 2;
        Logistic = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Logistic(i, j) = miu .* Logistic(i, j - 1) .* (1 - Logistic(i, j - 1));
            end
        end
        result = lb + Logistic .* (ub - lb);
        
    elseif label == 5
        % Sine
        sine = 2;
        Sine = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Sine(i, j) = (4 / sine) * sin(pi * Sine(i, j - 1));
            end
        end
        result = lb + Sine .* (ub - lb);
        
    elseif label == 6
        % Circle
        a = 0.5;
        b = 0.6;
        Circle = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Circle(i, j) = mod(Circle(i, j - 1) + a - b / (2 * pi) * sin(2 * pi * Circle(i, j - 1)), 1);
            end
        end
        result = lb + Circle .* (ub - lb);
        
    else
      
        result = lb + rand(N, dim) .* (ub - lb);
    end
end

%% 
function s = Bounds(s, Lb, Ub)
   
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);
    

    J = temp > Ub;
    temp(J) = Ub(J);
    
   
    temp(1:3) = round(temp(1:3));
    
    s = temp;
end

%% 
function Positions = initialization(SearchAgents_no, dim, ub, lb)
   
    
    Boundary_no = size(ub, 2); 
    
  
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    
    
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% 
function Y = normr(X)
   
    
    [m, n] = size(X);
    Y = zeros(m, n);
    
    for i = 1:m
        norm_row = norm(X(i, :));
        if norm_row == 0
            Y(i, :) = zeros(1, n);
        else
            Y(i, :) = X(i, :) / norm_row;
        end
    end
end

