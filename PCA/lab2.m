function main()
clc;
close all;
clear all;
Section1();
Section2();
Bonus();
end

function Section1()
disp("Executing section 1");
x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14];
y = [2.56,1.14,4.01,6.02,4.62,7.48,7.79,9.40,10.43,9.23,13.22,13.94,14.30,15.32,14.97];
% centre the dataset
xMean = mean(x);
yMean = mean(y);
for i = 1:length(x)
    y(i) = y(i) - yMean;
    x(i) = x(i) - xMean;
end
% Q1 we can assume that the dimensionality of the dataset is 1 as the
% best fit line modelled by the data is a straight line.

% compute the covariance matrix of x and y
covMatrix = cov(x, y);
[eigVec, diagEigVal] = eig(covMatrix);
disp("eignvectors");
disp(eigVec);
disp("eigenvalue diagonal matrix");
disp(diagEigVal);
% Q2: largest eigenvalue = 41.53, largest eignvector = [0.6896, 0.7242]
% There ia a relation between V and the spearhead of the datapoints.  V is
% positive and the datapoints move up and to the right on the plane.  This
% is also seen in the eigenvector which has 2 positive values,, indicating
% a move up the x and y axis in the positive direction.

%Randomly generated 100 sampled 2D matrix
z = randi(10, 10);
zMean = mean(z);
for i = 1:10
    for j = 1:10
        z(i, j) = z(i, j) - zMean(j);
    end
end
%TODO: fix this section to actually calculate dimensionality
disp("The randomly generated matrix");
disp(z);
zCov = cov(z);
[EV, DE] = eig(zCov);
disp("right eignvector");
disp(EV);
disp("eigenvalue diagonal matrix");
disp(DE);
% Q3: The true dimensionality (3-5) seems to be much lower than the original
% dimensionality (10).  We may be able to prioritize the first 2 or 3
% principal components of the matrix as they contain the largest variance.
end

function Section2()
disp("Executing section 2");
% Get training data
% TODO: fix below code to actually read files in folders
imColMat = readIn(320);
% create mean vector
meanVec = mean(imColMat, 2);
% reshape mean vector
reshapedMean = reshape(meanVec, [112, 92]);
%imshow(reshapedMean);
reshapedMean = (reshapedMean - min(reshapedMean, [], 'all')) / (max(reshapedMean, [], 'all') - min(reshapedMean, [], 'all'));
imwrite(reshapedMean, "mean.bmp");
dataMat = zeros(10304, 320);
for i = 1:320
    dataMat(:, i) = imColMat(:, i) - meanVec;
end
% calculate the covariance eigenvectors and eigenvalues
n = 320;
tMat = (1/n) * transpose(dataMat) * dataMat;
[rightEigVec, diagEigVal] = eig(tMat);
eigVal = eig(tMat);
covEigVec = dataMat * rightEigVec;
% get the 5 largest and 5 smallest eigenvalues and their corresponding
% eigenvectors
lgEigVec = zeros(10304, 5);
for i = 1:5
    lgEigVec(:, i) = covEigVec(:, i);
end
j = 1;
smEigVec = zeros(10304, 5);
for i = (length(eigVal) - 4):(length(eigVal))
    smEigVec(:, j) = covEigVec(:, i);
    j = j + 1;
end
% reshape the eigenvectors
% reshaping the largest eigenvectors
for i = 1:5
    reshapedEigVec = reshape(lgEigVec(:, i), [112, 92]);
    imshow(reshapedEigVec, []);
    filename = "Q5_" + i + "thEigen.bmp";
    reshapedEigVec = (reshapedEigVec - min(reshapedEigVec, [], 'all')) / (max(reshapedEigVec, [], 'all') - min(reshapedEigVec, [], 'all'));
    imwrite(reshapedEigVec, filename);
    %pause(5);
end
% reshaping the smallest eigenvectors
for i = 1:5
    reshapedEigVec = reshape(smEigVec(:, i), [112, 92]);
    imshow(reshapedEigVec, []);
    reshapedEigVec = (reshapedEigVec - min(reshapedEigVec, [], 'all')) / (max(reshapedEigVec, [], 'all') - min(reshapedEigVec, [], 'all'));
    filename = "Q6_" + i + "thEigen.bmp";
    imwrite(reshapedEigVec, filename);
    %pause(5);
end
% For the smallest eigenvectors, the images created upon reshaping the
% eigenvectors do not contain much, if any, information.

% svd decomposition section 
[U, D, V] = svd(dataMat);
lgSvdVec = zeros(10304, 5);
% calculate eigenvalues
D = (D.^2) / 320;
for i = 1:5
    lgSvdVec(:, i) = U(:, i);
end
% display and save 5 eigenfaces with largest eigenvalues
for i = 1:5
    reshapedSvdVec = reshape((lgSvdVec(:, i)), [112, 92]);
    imshow(reshapedSvdVec, []);
    reshapedSvdVec = (reshapedSvdVec - min(reshapedSvdVec, [], 'all')) / (max(reshapedSvdVec, [], 'all') - min(reshapedSvdVec, [], 'all'));
    filename = "Q7_" + i + "thEigenSVD.bmp";
    imwrite(reshapedSvdVec, filename);
    %pause(5);
end

% reconstruction
origImg = (imColMat(:, 1) - min(imColMat(:, 1), [], 'all')) / (max(imColMat(:, 1), [], 'all') - min(imColMat(:, 1), [], 'all'));
imwrite(reshape(origImg, [112, 92]), "Q9org.bmp");
reconImg = reconstruction(U, imColMat(:, 1), meanVec, 50);
reconImg = (reconImg - min(reconImg, [], 'all')) / (max(reconImg, [], 'all') - min(reconImg, [], 'all'));
imwrite(reconImg, "Q8rec.bmp");
reconImg = reconstruction(U, imColMat(:, 1), meanVec, 100);
reconImg = (reconImg - min(reconImg, [], 'all')) / (max(reconImg, [], 'all') - min(reconImg, [], 'all'));
imwrite(reconImg, "Q9rec.bmp");
end

function Bonus()
disp("Executing bonus");
% Get training data
% TODO: fix below code to actually read files in folders
imColMat = readIn(328);
% create mean vector
meanVec = mean(imColMat, 2);
% reshape mean vector
dataMat = zeros(10304, 328);
for i = 1:321
    dataMat(:, i) = imColMat(:, i) - meanVec;
end
% calculate the covariance eigenvectors and eigenvalues
n = 320;
% get the 5 largest and 5 smallest eigenvalues and their corresponding
% eigenvectors

% svd decomposition section 
[U, D, V] = svd(dataMat);
% calculate eigenvalues
D = (D.^2) / 328;

% reconstruction
origImg = (imColMat(:, 281) - min(imColMat(:, 281), [], 'all')) / (max(imColMat(:, 281), [], 'all') - min(imColMat(:, 281), [], 'all'));
imwrite(reshape(origImg, [112, 92]), "Q10org.bmp");
reconImg = reconstruction(U, imColMat(:, 281), meanVec, 50);
reconImg = (reconImg - min(reconImg, [], 'all')) / (max(reconImg, [], 'all') - min(reconImg, [], 'all'));
imwrite(reconImg, "Q10rec.bmp");
end

function imColMat = readIn(size)
folders = dir('training');
folders = folders(~ismember({folders.name},{'.','..'}));
subFolders = folders([folders.isdir]);
imColMat = zeros(10304, size);
j = 1;
for k = 1 : length(subFolders)
    if (subFolders(k).name == "s41" && size == 320)
        continue
    end
    cur_dr = ['training\' subFolders(k).name];
    images = dir(cur_dr);
    images = images(~ismember({images.name},{'.','..'}));
    for i=1 : length(images)
        im = imread([cur_dr '\' images(i).name]);
        % take only the first layer of the 3D matrix as they are all the
        % same
        im = im(:, :, 1);
        colVec = reshape(im, [], 1);
        imColMat(:, j) = colVec;
        j = j + 1;
    end
end
end

function reconImg = reconstruction(V, origImg, newMean, size)
reconV = zeros(10304, size);
for i = 1:size
    reconV(:, i) = V(:, i);
end
img = zeros(10304, 1);
for i = 1:10304
    img(i) = origImg(i) - newMean(i);
end
W = transpose(reconV) * img;
reconImg = reconV * W;
reconImg = reconImg + newMean;
reconImg = reshape(reconImg, [112, 92]);
imshow(reconImg, []);
end
