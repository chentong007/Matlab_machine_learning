    % Using PCA to evaluate raw input data
    % Classified data labels
    
    load('X1600');
    load('Te28');
    load('Lte28');
    X = X1600;  % Input data
    Y = [0 1 2 3 4 5 6 7 8 9]; % Output labels
    Xtrain = reshape(X,784,1600,10); % Create a 3D matrix for 10 datasets
    [row,col,stack] = size(Xtrain);
    
    u_j=zeros(row,1,stack); % preallocate dimensions
    A_j=zeros(row,col,stack);
    C_j=zeros(row,row,stack);
    Uqj=zeros(row,29,stack);
    Sqj=zeros(29,29,stack);
    f_j=zeros(29,1,stack);
    e_j=zeros(1,1,stack);
    
    % Trainning data set from X1600.mat
    for j = 0:stack-1
        u_j(:,:,j+1) = mean((Xtrain(:,:,j+1))')'; % Calculate mean values
        
        A_j(:,:,j+1) = Xtrain(:,:,j+1) - u_j(:,:,j+1);
        
        % Covariance matrix
        C_j(:,:,j+1) = (A_j(:,:,j+1))*(A_j(:,:,j+1))'/col;
        
        [Uqj(:,:,j+1),Sqj(:,:,j+1)] = eigs (C_j(:,:,j+1),29);
    end
    
    % Testing data of Te28.mat
    B = Te28;
    [row_test,col_test] = size(Te28);
    classified = zeros(col_test,1);
    for i= 1:col_test
        for j = 0:stack-1
        
        % Obtain principal components
        f_j(:,:,j+1) = (Uqj(:,:,j+1))' * (B(:,i) - u_j(:,:,j+1));
        
        X_j = (Uqj(:,:,j+1)) * f_j(:,:,j+1) + u_j(:,:,j+1);
        
        e_j(:,:,j+1) = norm(B(:,i)-X_j);
        end
        
        % Get index of minimum ej
        [~, index]= min(e_j);
        % Take minimum distance and classified
        classified(i) = index-1;
    end
    
    % Compare labels in Te28.mat with Lte28.mat
    miss_classified = nnz(classified-Lte28);
    
    % Image display first set of input data
    figure(1);
    for i=1:10
        hold on
        subplot(4,4,i)
        imshow(reshape(Xtrain(:,1,i),28,28),'InitialMagnification','fit');
    end
    
    % Image display after feature extraction
    figure(2);
    for i=1:10
        hold on
        subplot(4,4,i)
        imshow(reshape(A_j(:,1,i),28,28),'InitialMagnification','fit');
    end
    
    % Calculate accuracy
    accuracy = (col_test-miss_classified)/col_test;