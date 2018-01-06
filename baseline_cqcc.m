%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASVspoof 2017 CHALLENGE:
% Audio replay detection challenge for automatic speaker verification anti-spoofing
% 
% http://www.spoofingchallenge.org/
% 
% ====================================================================================
% Matlab implementation of the baseline system for replay detection based
% on constant Q cepstral coefficients (CQCC) features + Gaussian Mixture Models (GMMs)
% ====================================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));

% set paths to the wave files and protocols
pathToDatabase = '/mnt/speechlab/users/hedi7/data/ASVspoof2017/'
trainProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_train.trn.txt');
devProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_dev.trl.txt');
evaProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_eval_v2_key.trl.txt');


% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

fmax = 8000;
fmin = 7000;
B=256;
d=256;
cf=29;

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);

    genuineFeatureCell{i} = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD');
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);

    spoofFeatureCell{i} = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD');
end
disp('Done!');

%% GMM training

% train GMM for GENUINE data
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');

% train GMM for SPOOF data
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');

%% Scoring

% score for dev data
score(devProtocolFile)

% score for eval data
score(evaProtocolFile)


%% Feature extraction and scoring of development data

function score(protocalFile)
    % read development protocol
    fileID = fopen(protocalFile);
    protocol = textscan(fileID, '%s%s%s%s%s%s%s');
    fclose(fileID);

    % get file and label lists
    filelist = protocol{1};
    labels = protocol{2};

    % process each development trial: feature extraction and scoring
    scores = zeros(size(filelist));
    disp('Computing scores for development trials...');
    parfor i=1:length(filelist)
        filePath = fullfile(pathToDatabase,'ASVspoof2017_dev',filelist{i});
        [x,fs] = audioread(filePath);
        % featrue extraction
        x_cqcc = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD');

        %score computation
        llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
        llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
        % compute log-likelihood ratio
        scores(i) = llk_genuine - llk_spoof;
    end
    disp('Done!');

    % compute performance
    [Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
    EER = rocch2eer(Pmiss,Pfa) * 100; 
    fprintf('EER is %.2f\n', EER);
end
