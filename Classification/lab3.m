function Main()
clc;
clear all;
close all;
Sections();
end

function Sections()
disp("running lab 3");
data = zeros(10304, 320);
data = readIn();
dataMat = createDataMat(data);
[U, D, V] = svd(dataMat);
eigVals = svd(dataMat);
disp("displaying eigenvalues");
bar(eigVals);
pause(5);
% Q1: I would choose K by examining the graph and looking for a point near
% where the graph starts to flatten out.  This point is where I would get
% K from.
weights = contribEig(dataMat, U);
weightedMean = mean(weights, 2);
% loading arctichare.png
aHare = imread("arctichare.png");
aHare = aHare(:, :, 1);
resizedHare = imresize(aHare, [112, 92]);
hareCol = reshape(resizedHare, [], 1);
U50 = zeros(10304, 50);
for i = 1:50
    U50(:, i) = U(:, i);
end
meanVec = mean(data, 2);
wHare = transpose(U50) * (double(hareCol) - meanVec);
dHare = norm(wHare - weightedMean);
wTest1 = transpose(U50) * (data(:, 11) - meanVec);
dTest1 = norm(wTest1 - weightedMean);
% Q2
disp("hare");
disp(dHare);
disp("test1");
disp(dTest1);

% Q3
wTest2 = transpose(U50) * (data(:, 23) - meanVec);
dTest2 = norm(wTest2 - weightedMean);
wTest3 = transpose(U50) * (data(:, 47) - meanVec);
dTest3 = norm(wTest3 - weightedMean);
wTest4 = transpose(U50) * (data(:, 8) - meanVec);
dTest4 = norm(wTest4 - weightedMean);

disp("test2");
disp(dTest2);
disp("test3");
disp(dTest3);
disp("test4");
disp(dTest4);

% set the threshold at 6.000e+03
threshold = 6000;
disp("Threshold: " + threshold);

% Section 4: face recognition
weightedMeans = meanPerImg(weights);
reconImg1 = (U50 * weightedMeans(:, 1)) + meanVec;
imshow(reshape(reconImg1, [112, 92]), []);
reconImg1 = (reconImg1 - min(reconImg1, [], 'all')) / (max(reconImg1, [], 'all') - min(reconImg1, [], 'all'));
imwrite(reshape(reconImg1, [112, 92]), "Q4_1strecon.bmp");
reconImg40 = (U50 * weightedMeans(:, 35)) + meanVec;
imshow(reshape(reconImg40, [112, 92]), []);
reconImg40 = (reconImg40 - min(reconImg40, [], 'all')) / (max(reconImg40, [], 'all') - min(reconImg40, [], 'all'));
imwrite(reshape(reconImg40, [112, 92]), "Q4_40threcon.bmp");
% Q4: the images do not resemble their original images at all  however,
% they do resemble eachother.  This is likely due to the fact that the
% weights for both the 1st and 40th image set are very similar and we are
% using the same values for V and E[X].

testImgs = readInTest();
meanTest = mean(testImgs, 2);
[Ut, Dt, Vt] = svd(testImgs);
testWeights = zeros(50, 80);
Ut50 = zeros(10304, 50);
for i = 1:50
    Ut50(:, i) = Ut(:, i);
end
for i = 1:80
    testWeights(:, i) = transpose(Ut50) * (testImgs(:, i) - meanTest);
end
% Q5
eucNorm = labels(testWeights, weightedMeans);
disp("Q5 accuracies");
disp(eucNorm);

% Q6
knn = KNN(testWeights, weights);
disp("Q6 KNN");
disp(knn);
% Q6: 

% Q7
wMyIm = Q7(wHare, weightedMean, U50, meanVec);

minMyIm = 10000000;
whichIm = 0;
for i = 1:320
    nrm = norm(wMyIm - weights(:, i));
    if ((nrm < minMyIm))
        whichIm = i;
        minMyIm = nrm;
    end
end
disp("shortest euclidean distance to my test image");
disp(minMyIm);
disp("Closest image to my test image");
imshow(reshape(data(:, whichIm), [112, 92]), []);

% Bonus question Q8
Knn3 = bonus(wMyIm, weights, 3);
Knn5 = bonus(wMyIm, weights, 5);
Knn7 = bonus(wMyIm, weights, 7);
disp("Knn3: " + Knn3);
disp("Knn5: " + Knn5);
disp("Knn7: " + Knn7);
disp("Mode of Knn3: " + mode(Knn3));
disp("Mode of Knn5: " + mode(Knn5));
disp("Mode of Knn7: " + mode(Knn7));
end

function whichIm = bonus(wMyIm, weights, size)
whichIm = zeros(1, size);
eucNorms = zeros(1, 320);
for i = 1:320
    eucNorms(i) = norm(wMyIm - weights(:, i));
end
for i = 1:size
    minMyIm = 10000000;
    for j = 1:length(eucNorms)
        if (eucNorms(j) < minMyIm)
            minMyIm = eucNorms(j);
            whichIm(i) = j;
        end
    end
    eucNorms(whichIm(i)) = [];
end
end

function wMyIm = Q7(wHare, wMean, V, meanVec)
myIm = imread("myTest.jpg");
myIm = myIm(:, :, 1);
myIm = reshape(myIm, [], 1);
myIm = double(myIm);
wMyIm = transpose(V) * (myIm - meanVec);
disp(wMyIm);
disp(wHare);
disp(wMean);
end

function knn = KNN(wTests, wTrains)
knn = zeros(1, 80);
for i = 1:80
    temp = zeros(1, 320);
    for j = 1:320
        temp(:, j) = norm(wTests(:, i) - wTrains(:, j));
    end
    knn(:, i) = min(temp);
end
%for i = 1:80
%    knn(:, i) = mod(knn(:, i), 1);
%end
end

function eucNorm = labels(tWeights, wMeans)
eucNorm = zeros(1, 80);
for i = 1:80
    temp = zeros(1, 40);
    for j = 1:40
        temp(:, j) = norm(tWeights(:, i) - wMeans(:, j));
    end
    eucNorm(:, i) = min(temp);
end
end

function weights = meanPerImg(meanMatrix)
weights = zeros(50, 40);
k = 1;
for i = 1:8:320
    n = 1;
    imgWeights = zeros(50, 8);
    for j = i:(i + 7)
        imgWeights(:, n) = meanMatrix(:, j);
        n = n + 1;
    end
    weights(:, k) = mean(imgWeights, 2);
    k = k + 1;
end
end

function weights = contribEig(dataMatrix, eigenVec)
weights = zeros(50, 320);
V = zeros(10304, 50);
for i = 1:50
    V(:, i) = eigenVec(:, 1);
end
for i = 1:320
    weights(:, i) = transpose(V) * dataMatrix(:, i);
end
end

function data = readIn()
folders = dir('training');
folders = folders(~ismember({folders.name},{'.','..'}));
subFolders = folders([folders.isdir]);
data = zeros(10304, 320);
j = 1;
for k = 1 : length(subFolders)
    cur_dr = ['training\' subFolders(k).name];
    images = dir(cur_dr);
    images = images(~ismember({images.name},{'.','..'}));
    for i=1 : length(images)
        im = imread([cur_dr '\' images(i).name]);
        % take only the first layer of the 3D matrix as they are all the
        % same
        im = im(:, :, 1);
        colVec = reshape(im, [], 1);
        data(:, j) = colVec;
        j = j + 1;
    end
end
end

function data = readInTest()
folders = dir('testing');
folders = folders(~ismember({folders.name},{'.','..'}));
subFolders = folders([folders.isdir]);
data = zeros(10304, 80);
j = 1;
for k = 1 : length(subFolders)
    cur_dr = ['testing\' subFolders(k).name];
    images = dir(cur_dr);
    images = images(~ismember({images.name},{'.','..'}));
    for i=1 : length(images)
        im = imread([cur_dr '\' images(i).name]);
        % take only the first layer of the 3D matrix as they are all the
        % same
        im = im(:, :, 1);
        colVec = reshape(im, [], 1);
        data(:, j) = colVec;
        j = j + 1;
    end
end
end

function dataMatrix = createDataMat(dataIn)
meanVec = mean(dataIn, 2);
dataMatrix = zeros(10304, 320);
for i = 1:320
    dataMatrix(:, i) = dataIn(:, i) - meanVec;
end
end