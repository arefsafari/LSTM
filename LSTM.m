%LSTM by Aref Safari

datafolder = PathToDatabase;
ads0 = audioDatastore(fullfile(datafolder,"AQI,Dust Storm, Macky Glass, Sunspot"));
metadataTrain = readtable(fullfile(datafolder,"train.tsv"),"FileType","text");
metadataDev = readtable(fullfile(datafolder,"dev.tsv"),"FileType","text");
metadata = [metadataTrain;metadataDev];
head(metadata)
csvFiles = metadata.path;
adsFiles = ads0.Files;
adsFiles = cellfun(@HelperGetFilePart,adsFiles,'UniformOutput',false);
[~,indA,indB] = intersect(adsFiles,csvFiles);
adsTrain = subset(ads0,indA);
CC = repmat(centroid,windowLength,1);
CC = CC(:);
EE = repmat(signalEnergy,windowLength,1);
EE = EE(:);
flags2 = repmat(isSpeechRegion,windowLength,1);
flags2 = flags2(:);
figure
subplot(3,1,1)
plot(timeVector, CC(1:numel(audio)), ...
     timeVector, repmat(T_C,1,numel(timeVector)), "LineWidth",2)
xlabel("Time (s)")
ylabel("Normalized Centroid")
legend("Centroid","Threshold")
title("Spectral Centroid")
grid on

subplot(3,1,2)
plot(timeVector, EE(1:numel(audio)), ...
     timeVector, repmat(T_E,1,numel(timeVector)),"LineWidth",2)
ylabel("Normalized Data")
legend("Train","Threshold")
title("Test")
grid on

subplot(3,1,3)
plot(timeVector, audio, ...
     timeVector,flags2(1:numel(audio)),"LineWidth",2)
ylabel("X")
legend("X","Y")
title("Legend")
grid on
ylim([-1 1.1])
activeSegment = 1;
isSegmentsActive = zeros(1,numel(startIndices));
isSegmentsActive(1) = 1;
for index = 2:numel(startIndices)
    if startIndices(index) <= endIndices(activeSegment)
        % Current segment intersects with previous segment
        if endIndices(index) > endIndices(activeSegment)
           endIndices(activeSegment) =  endIndices(index);
        end
    else
        % New speech segment detected
        activeSegment = index;
        isSegmentsActive(index) = 1;
    end
end
numSegments = sum(isSegmentsActive);
segments = cell(1,numSegments);
limits = zeros(2,numSegments);
speechSegmentsIndices  = find(isSegmentsActive);
for index = 1:length(speechSegmentsIndices)
    segments{index} = audio(startIndices(speechSegmentsIndices(index)): ...
                            endIndices(speechSegmentsIndices(index)));
    limits(:,index) = [startIndices(speechSegmentsIndices(index)); ...
                       endIndices(speechSegmentsIndices(index))];
end

figure

plot(timeVector,audio)
hold on
myLegend = cell(1,numel(segments) + 1);
myLegend{1} = "Original Audio";
for index = 1:numel(segments)
    plot(timeVector(limits(1,index):limits(2,index)),segments{index});
    myLegend{index+1} = sprintf("Output Audio Segment %d",index);
end
xlabel("Time (s)")
ylabel("Audio")
grid on
legend(myLegend)

win = hamming(0.03*Fs,"periodic");
overlapLength = round(0.75*numel(win));
featureParams = struct("SampleRate",Fs, ...
                 "Window",win, ...
                 "OverlapLength",overlapLength);

% Define the LSTM Network Architecture
layers = [ ...
    sequenceInputLayer(size(featuresTrain{1},1))
    bilstmLayer(50,"OutputMode","sequence")
    dropoutLayer(0.1)
    bilstmLayer(50,"OutputMode","last")
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

miniBatchSize = 128;
validationFrequency = floor(numel(genderTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    "MaxEpochs",4, ...
    "MiniBatchSize",miniBatchSize, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",2,...
    'ValidationData',{featuresValidation,categorical(genderValidation)}, ...
    'ValidationFrequency',validationFrequency);

net = trainNetwork(featuresTrain,categorical(genderTrain),layers,options);

figure
cm = confusionchart(categorical(genderValidation),valPred,'title','Validation Set Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
sequencePerFile = zeros(size(valSegmentsPerFile));
valSequencePerSegmentMat = cell2mat(valSequencePerSegment);
idx = 1;
for ii = 1:numel(valSegmentsPerFile)
    sequencePerFile(ii) = sum(valSequencePerSegmentMat(idx:idx+valSegmentsPerFile(ii)-1));
    idx = idx + valSegmentsPerFile(ii);
end

numFiles = numel(adsVal.Files);
actualGender = categorical(adsVal.Labels);
predictedGender = actualGender;      
scores = cell(1,numFiles);
counter = 1;
cats = unique(actualGender);
for index = 1:numFiles
    scores{index}      = valScores(counter: counter + sequencePerFile(index) - 1,:);
    m = max(mean(scores{index},1),[],1);
    if m(1) >= m(2)
        predictedGender(index) = cats(1);
    else
        predictedGender(index) = cats(2); 
    end
    counter = counter + sequencePerFile(index);
end

% Visualize the confusion matrix on the predictions.
figure
cm = confusionchart(actualGender,predictedGender,'title','Validation Set Accuracy - Max Rule');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';