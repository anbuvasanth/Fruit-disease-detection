[fname, path]=uigetfile('.jpg','provide an Image for testing');
fname=strcat(path, fname);
img=imread(fname);
img = imresize(img,[256,256]);

figure(1),imshow(img);
title('Original image');

% Enhance Contrast
img = imadjust(img,stretchlim(img));
figure, imshow(img);title('Contrast Enhanced');

% % converting the image to L*a*b form
cform = makecform('srgb2lab');
lab_he = applycform(img,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 2;
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = img;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end



% figure(),
% imshow(segmented_images{3});title('Cluster 3');

% Feature Extraction
% x = inputdlg('Enter the cluster no. containing the disease affected  part only:');
% i = str2double(x);

% Convert to grayscale if image is RGB
count0=0;
count1=0;
figure(), imshow(segmented_images{1});title('Cluster 1');
figure(), imshow(segmented_images{2});title('Cluster 2');
% figure(), imshow(segmented_images{3});title('Cluster 3');
for loop=1:2
     
    
    seg_img = segmented_images{loop};
    if ndims(seg_img) == 3
       img = rgb2gray(seg_img);
    end
    black = im2bw(seg_img,graythresh(seg_img));

    
    m = size(seg_img,1);
    n = size(seg_img,2);

    zero_image = zeros(m,n);
 
glcms = graycomatrix(img);

% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
% Smoothness = 1-(1/(1+a));
% Kurtosis = kurtosis(double(seg_img(:)));
% Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
feat_disease = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance,IDM];
load('TrainData.mat')

% Put the test features into variable 'test'
test = feat_disease;

X = Trainvalue';    
Y = Trainresult'; 
 Mdl = fitcknn(X,Y);
% Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
CVKNNMdl = crossval(Mdl);
classError = kfoldLoss(CVKNNMdl);

class = predict(Mdl, test);


if class==1
%     disp('healthy')
    count1=count1+1;
elseif class==0
%     disp('bacterial disease')
    count0=count0+1;
    figure(),imshow(seg_img);title('affected part');
end

end

disp(count1)
disp(count0)
if count1==2
    helpdlg(' Healthy ');
elseif (count1==1 && count0==1)
    helpdlg(' Grey Mildew Disease '); 
elseif count0==2
    helpdlg(' bacterial disease ');
    
end
    


% cc = bwconncomp(seg_img,6);
% diseasedata = regionprops(cc,'basic');
% A1 = diseasedata.Area;
% sprintf('Area of the disease affected region is : %g%',A1);


% load('Accuracy.mat')
% 
% % Put the test features into variable 'test'
% test = feat_disease;
% 
% X = input';    
% Y = target'; 
% Mdl = fitcknn(X,Y);
% % Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
% 
% class = predict(Mdl, test);
% 
% disp(class)
% if class==1
%     disp('healthy')
% elseif class==0
%     disp('bacterial disease')
% end

