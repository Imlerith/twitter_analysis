%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%SUPPORT VECTOR MACHINES FOR TWITTER ANALYSIS%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load classCatList 
% load PCs
% load svmmdl
% load CVMdl

%Create the feature vector and pre-allocate for feature matrices
featListCrct = cellstr(featList);
Xmat         = zeros(length(cltwtList),length(featListCrct));
Ymat         = classCatList';
UXmat        = zeros(length(uncltwtList),length(featListCrct));
emptyTwt     = [];

% %Choose randomly which neutrals to keep
% IndNeut     = find(classCatList(:) == 0);
% IndNeg      = find(classCatList(:) == -1);
% IndPos      = find(classCatList(:) == 1);
% IndNeutKeep = randsample(IndNeut,6000); %arbitrarily take 40000
% IndTot      = sort([IndNeutKeep;IndNeg;IndPos]);

%%%%%%%%%%%%%%%%%Construct the training matrix X%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(cltwtList)
    if isempty(cltwtList{i})
        emptyTwt = [emptyTwt,i]; %remove these rows from the matrix!
        continue        
    end
        
    cltwt = cellstr(cltwtList{i});
    [intSet,indint,~] = intersect(featListCrct,cltwt);
    
    Xmat(i,indint) = 1;
    disp(i)
end
Xmat(emptyTwt,:) = [];
Ymat(emptyTwt,:) = [];

% XmatReduced = Xmat(IndTot,:);
% YReduced    = classCatList(IndTot);

%Remove entries with all zeros 
indNonNull = [];
for i = 1:size(Xmat,1)
    if sum(Xmat(i,:)) ~= 0;
        indNonNull = [indNonNull,i];
    end
end
XmatCleaned = Xmat(indNonNull,:);
YmatCleaned = Ymat(indNonNull);

XmatCleaned = Xmat;
YmatCleaned = Ymat;

clear Ymat Xmat


%%%%%%%%%%%%%%%%%Construct the test X-matrix%%%%%%%%%%%%%%%%%%%%%%%
emptyTwtUncl = [];
for i = 1:length(uncltwtList)
    if isempty(uncltwtList{i})
        emptyTwtUncl = [emptyTwtUncl,i];
        continue        
    end
        
    uncltwt = cellstr(uncltwtList{i});
    [intSet,indint,~] = intersect(featListCrct,uncltwt);
    
    UXmat(i,indint) = 1;
    disp(i)
end
UXmat(emptyTwtUncl,:) = [];
testCatList(emptyTwtUncl) = [];


%Remove entries with all zeros
indNonNullUncl = [];
for i = 1:size(UXmat,1)
    if sum(UXmat(i,:)) ~= 0;
        indNonNullUncl = [indNonNullUncl,i];
    end
end
UXmatCleaned = UXmat(indNonNullUncl,:);
testCatList = testCatList(indNonNullUncl);
clear UXmat


%%%%%%%%%%%%%%%%%%%Merge two matrices%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AllXmat        = [Xmat;UXmat];
AllXmatCleaned = [XmatCleaned;UXmatCleaned];

% XmatInt = int8(Xmat);
% classCatList(emptyTwt) = [];
% classCatList = double(classCatList);


% Reduce the dimension of feature matrix
% [~,PCs,~,~,ExpldVar,~]        = pca(Xmat);
% [~,AllPCs,~,~,AllExpldVar,~]  = pca(AllXmat);
[~,AllPCs,~,~,AllExpldVar,~] = pca(AllXmatCleaned);

%%%%%%%%%%% Alternative way to reduce (multidimensional scaling)%%%%%%%%%%%
n  = size(AllXmatCleaned,1);
N  = sum(sum(AllXmatCleaned));
P  = AllXmatCleaned/N; %correspondence matrix
r  = sum(P,2); %row total (row mass)
c  = sum(P,1); %column total (column mass)
Dr = diag(r); %row diagonal matrix
Dc = diag(c); %column diagonal matrix
Z  = sqrt((Dr)^(-1))*(P-r*c)*sqrt((Dc)^(-1));
%non-trivial eigenvalues (maximum number of dimensions)
if n >= p,
    q = sum(g)-p; 
else
    q = p-n;
end
[U,S,V] = svd(Z); %singular values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ExplVarCumul     = cumsum(AllExpldVar);
LargerThanThresh = find(ExplVarCumul > 95);
NumOfVars        = LargerThanThresh(1);

%%%%%%%%%%%%%%%%%%%%%Train multiclass SVM classifier%%%%%%%%%%%%%%%%%%%
NTrain  = length(classCatList); 
XTrain  = AllPCs(1:NTrain,1:NumOfVars);
XTest   = AllPCs(NTrain+1:end,1:NumOfVars);

parpool(5)
options = statset('UseParallel',1);
SvmMdl  = fitcecoc(XmatCleaned,YmatCleaned,'Options',options);
CVMdl   = crossval(SvmMdl,'KFold',5,'Options',options); %5-fold cross-validation
GenLoss = kfoldLoss(CVMdl);

%%%%%%%%%%%%%%%%%%%%%Predict labels for unclassified tweets%%%%%%%%%%%%%%%%
CatPred = predict(SvmMdl,UXmatCleaned);

countCorrect = 0;
for i = 1:length(CatPred)
    
    if CatPred(i) == testCatList(i);
        countCorrect = countCorrect + 1;
    end
    
end

percTotCorrect = countCorrect/length(CatPred);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%CONFUSION MATRIX CALCULATION%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NN     = 0;
PN     = 0;
NtrN   = 0;
NP     = 0;
PP     = 0;
NtrP   = 0;
NNtr   = 0;
PNtr   = 0;
NtrNtr = 0;
for i = 1:length(CatPred)
    
    %actual negatives
    if CatPred(i) == -1 && testCatList(i) == -1
        NN = NN + 1;
        
    elseif CatPred(i) == 1 && testCatList(i) == -1
        PN = PN + 1;
        
    elseif CatPred(i) == 0 && testCatList(i) == -1
        NtrN = NtrN + 1;
        
    %actual positives
    elseif CatPred(i) == -1 && testCatList(i) == 1
        NP = NP + 1;
        
    elseif CatPred(i) == 1 && testCatList(i) == 1
        PP = PP + 1;
        
    elseif CatPred(i) == 0 && testCatList(i) == 1
        NtrP = NtrP + 1;
        
    %actual neutrals
    elseif CatPred(i) == -1 && testCatList(i) == 0
        NNtr = NNtr + 1;
        
    elseif CatPred(i) == 1 && testCatList(i) == 0
        PNtr = PNtr + 1;
        
    elseif CatPred(i) == 0 && testCatList(i) == 0
        NtrNtr = NtrNtr + 1;
        
    end
    
end

Rows     = {'PredNeg';'PredPos';'PredNeutr'};
ActNeg   = [NN;PN;NtrN];
ActPos   = [NP;PP;NtrP];
ActNeutr = [NNtr;PNtr;NtrNtr];

ConfTab  = table(ActNeg,ActPos,ActNeutr,'RowNames',Rows)

ConfMat  = table2array(ConfTab)

%Compute Precision, Recall, F-score
RECALL    = ConfMat(1,1)/sum(ConfMat(1,:)); %aka SENSITIVITY 
PRECISION = ConfMat(1,1)/sum(ConfMat(:,1)); %aka PPV (positive predictive value)
FSCORE    = 2*((PRECISION*RECALL)/(PRECISION+RECALL)) 

%Confusion matrix in terms of TP-FP-FN-TN
ConfMat2  = [ConfMat(1,1),sum(ConfMat(1,2:end));
             sum(ConfMat(2:end,1)),sum(sum(ConfMat(2:end,2:end)))]


filename = '20160720_svm_twtr_autos_predict.mat';
save(filename,'-v7.3')

percNeg  = sum(classCatList(:) == -1)/length(classCatList)
percPos  = sum(classCatList(:) == 1)/length(classCatList)
percNeut = sum(classCatList(:) == 0)/length(classCatList)

load('featfrqs.mat')
load('featwrds.mat')
featwrds       = cellstr(featwrds);
[featfrqsSrtd,featInd] = sort(featfrqs,'descend');
featwrdsSrtd = featwrds(featInd);
bar(featfrqsSrtd)
set(gca,'XTick',1:100:2237,'XTickLabel',featwrdsSrtd(1:100:2237),'Ticklength', [0 0], 'FontSize',12, 'FontWeight', 'bold');
ax = gca;
ax.XTickLabelRotation = -45; 
ylabel('Frequency')




