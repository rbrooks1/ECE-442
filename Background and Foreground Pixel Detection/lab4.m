function main()
clc;
clear all;
close all;
thresholding();
MOG();
Q5();
end

function thresholding()
v1 = VideoReader("streetGray.mp4");
vFrames = v1.numFrames;

meanFrame = 0;
curSum = 0;
vid = {};
for i = 1:vFrames
    curFrame = read(v1, i);
    vid{end + 1} = im2single(curFrame);
    
    if (i == 1)
        curSum = double(curFrame);
    else
        curSum = meanFrame + double(curFrame);
    end
end

meanFrame = curSum / vFrames;
meanFrameGray = (meanFrame - min(meanFrame, [], 'all')) / (max(meanFrame, [], 'all') - min(meanFrame, [], 'all'));
imwrite(meanFrameGray, "Q1Mean.bmp");
% developed with help from https://www.mathworks.com/matlabcentral/answers/178819-i-have-written-a-code-for-adaptive-thresholding-in-background-subtraction-can-anyone-tell-me-what-i
foreground = abs(vid{420} - meanFrame);
optimum(foreground);
end

function optimum(foreground)
thresh = graythresh(foreground);
bIm = im2bw(foreground, thresh);
bIm = (bIm - min(bIm, [], 'all')) / (max(bIm, [], 'all') - min(bIm, [], 'all'));
imwrite(bIm, "Q1Optimum.bmp");
disp("Optimum threshold: " + thresh);
bImLess = im2bw(foreground, thresh - 0.1);
bImGreat = im2bw(foreground, thresh + 0.1);
subplot(1, 3, 1);
imshow(bIm, []);
title("optimum image");
subplot(1, 3, 2);
imshow(bImLess, []);
title("image less optimum threshold");
subplot(1, 3, 3);
imshow(bImGreat, []);
title("image great optimum threshold");
end

function MOG()
Y = VideoReader("streetGray.mp4");
K = 5;
iter = 600;
X = zeros(240, 1);
for i = 1:240
    curFrame = read(Y, i);
    X(i, :) = curFrame(360, 640);
end
% Q2

GMModel = fitgmdist(X,K,'RegularizationValue', 0.1,'Start',...
        'randSample','Options',statset('Display','off','MaxIter',iter,...
        'TolFun',1e-6));
disp(GMModel);

% Q3
frame420 = read(Y, 420);
pixel420 = frame420(360, 640);
x = double(pixel420);
interval = (x-3):0.0001:(x+3);
PDF_x = pdf(GMModel,interval');
probability5 = trapz(interval,PDF_x);

disp("Minimum Threshold for Q3: " + probability5);

%Q4
% repeat Q3 for K = 1 and K = 3 and compare
K = 1;
GMModel = fitgmdist(X,K,'RegularizationValue', 0.1,'Start',...
        'randSample','Options',statset('Display','off','MaxIter',iter,...
        'TolFun',1e-6));
interval = (x-3):0.0001:(x+3);
PDF_x = pdf(GMModel,interval');
probability1 = trapz(interval,PDF_x);

K = 3;
GMModel = fitgmdist(X,K,'RegularizationValue', 0.1,'Start',...
        'randSample','Options',statset('Display','off','MaxIter',iter,...
        'TolFun',1e-6));
interval = (x-3):0.0001:(x+3);
PDF_x = pdf(GMModel,interval');
probability3 = trapz(interval,PDF_x);

disp("minimum threshold for K = 1: " + probability1);
disp("minimum threshold for K = 3: " + probability3);
disp("minimum threshold for K = 5: " + probability5);
end

function Q5()
K = 5;
v1 = VideoReader("streetGray.mp4");
xCentre = v1.Width / 2;
yCentre = v1.Height / 2;
threshold = 0.0001;
xTopLeft = xCentre - 200;
yTopLeft = yCentre - 150;
frame420 = read(v1, 420);

% get all pixels
disp("getting pixels");
X = zeros(120000, 240);
for i = 1:240
    curFrame = read(v1, i);
    p = 1;
    for j = xTopLeft:xTopLeft + 399
        for k = yTopLeft:yTopLeft + 299
            X(p, i) = curFrame(k, j);
            p = p + 1;
        end
    end
    clear curFrame;
end
clear p;
disp("getting intervals");
% calculate intervals for each pixel in testing
intervals = zeros(120000, 10001);
p = 1;
l = 1;
pix420 = zeros(120000, 1);
frame420Box = zeros(300, 400);
for i = xTopLeft:xTopLeft + 399
    o = 1;
    for j = yTopLeft:yTopLeft + 299
        pixel = double(frame420(j, i));
        frame420Box(o, l) = pixel;
        pix420(p) = pixel;
        intervals(p, :) = (pixel-0.5):0.0001:(pixel+0.5);
        p = p + 1;
        o = o + 1;
        clear pixel;
    end
    l = l + 1;
end
clear p; clear l; clear o;
iter = 600;
disp("starting calc");
% calculate foreground and background pixels.
for p = 1:3
     foreground = zeros(300, 400);
     foreCol = zeros(120000, 1);
     background = zeros(300, 400);
     backCol = zeros(120000, 1);
     n = 1;
     for i = 1:120000
         disp(i);
         GMModel = fitgmdist(transpose(X(i, :)),K,'RegularizationValue', 0.1,'Start',...
             'randSample','Options',statset('Display','off','MaxIter',iter,...
             'TolFun',1e-6));
         PDF_x = pdf(GMModel,intervals(i, :)');
         probability = trapz(intervals(i, :),PDF_x);
         if (probability < threshold)
             foreCol(i) = pix420(i);
         else
             backCol(i) = pix420(i);
         end
         clear GMModel;
         clear PDF_x;
         clear probability;
     end
     clear n;
     n = 1;
     for i = 1:399
         if (n == 1)
            n = 1;
         else
            n = n + 1;
         end
         for j = 1:299
             foreground(j, i) = foreCol(n);
             background(j, i) = backCol(n);
             n = n + 1;
         end
     end
     clear n;
     foreFile = "thresh" + threshold + "_foreground.bmp";
     backFile = "thresh" + threshold + "_background.bmp";
     foregroundTest = frame420Box;
     foregroundTest = abs(foregroundTest - background);
     foregroundTest = (foregroundTest - min(foregroundTest, [], 'all')) / (max(foregroundTest, [], 'all') - min(foregroundTest, [], 'all'));
     imwrite(foregroundTest, "thresh" + threshold + "_foregroundTest.bmp");
     foreground = (foreground - min(foreground, [], 'all')) / (max(foreground, [], 'all') - min(foreground, [], 'all'));
     imwrite(foreground, foreFile);
     background = (background - min(background, [], 'all')) / (max(background, [], 'all') - min(background, [], 'all'));
     imwrite(background, backFile);
     clear foreFile; clear backFile; clear background; clear foreground;
     clear backCol; clear foreCol;
     threshold = threshold * 10;
end
end