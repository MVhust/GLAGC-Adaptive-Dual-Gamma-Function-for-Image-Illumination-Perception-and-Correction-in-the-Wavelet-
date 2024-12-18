
clc;
clear all;
close all;


input_img=imread('benchmark\q14.bmp');
input_img = imresize(input_img,[512,512]);
input_img = double(input_img);
IMAX = max(input_img(:));
input_img = uint8(IMAX*power(input_img/IMAX,1.0));
figure 
title("max");
imshow(input_img);
Output_img = uint8(V_adptive(input_img));
figure
imshow(Output_img);
title("V");

function param= GSLF_function(cA)
hist_cA = imhist(mat2gray(cA))/(512*512);
hist_cA = mat2gray(hist_cA);

cdf_cA = cumsum(hist_cA)/sum(hist_cA);
x_cols = 1:255/512:255; 
X =repmat(x_cols,512,1); % 人为构造的理想分布的图像
X = log(X+1);
[LL_1,LH,HL,LL] = dwt2(X,'haar');
hist_LIX = imhist(mat2gray(LL_1));
cdf_LIX = cumsum(hist_LIX)/sum(hist_LIX);
 PDF_NEW =power(hist_cA,1-(cdf_cA-cdf_LIX));
cdf_NEW = cumsum(PDF_NEW)/sum(PDF_NEW);
param = sum(abs((cdf_cA-cdf_NEW)))/sum(cdf_cA);
end
%% 算法主体
function return_img= V_adptive(img)
R_channel = double(img(:,:,1));
G_channel =double(img(:,:,2));
B_channel =double(img(:,:,3)); % RGB三个通道，mat2gray归一化处理（可有可无，仅为了方便）
%% V channel
[H,S,V] = rgb2hsv(img); 
I1 = V;  % 取V通道
tic
I1=log(I1+1); % 取对数域
[LL,LH,HL,HH] = dwt2(I1,'haar');  % 小波变换
I_max = max(LL(:));
GSLF = GSLF_function(LL); % 求全局特征值
Global_gama = 8.224*GSLF*GSLF-5.534*GSLF+1.093; % 求最佳全局校正值
LH=LH.*power(I_max,1-Global_gama)*Global_gama.*power(LL+0.01,Global_gama-1); 
HL=HL.*power(I_max,1-Global_gama)*Global_gama.*power(LL+0.01,Global_gama-1);
HH=HH.*power(I_max,1-Global_gama)*Global_gama.*power(LL+0.01,Global_gama-1);  % 线性抑制，+0.01是为了防止分母出现0
%  spatial_correction
SIGMA1=10;
SIGMA2=50;
SIGMA3=100;
HSIZE=size(LL);
F1 = fspecial('gaussian',HSIZE,SIGMA1);
F2 = fspecial('gaussian',HSIZE,SIGMA2);
F3 = fspecial('gaussian',HSIZE,SIGMA3);
gaus1= imfilter(LL, F1, 'replicate');
gaus2= imfilter(LL, F2, 'replicate');
gaus3= imfilter(LL, F3, 'replicate');
SLDF = double(gaus1/3)+double(gaus2/3)+double(gaus3/3);  % 求空间分布SLDF
m=mean(LL(:));
I_max=max(max(LL));
% global_correction
global_gama = Global_gama
spitial_gama=power(m/I_max,2*(m-SLDF)/I_max);
gama = spitial_gama*global_gama; % 自适应伽马值
out=I_max*power(LL/I_max,gama); % 自适应校正作用于LL
im1=idwt2(out,LH,HL,HH,'haar');
im1=exp(im1)-1;
delta =0.005* im1./V;
R_out = R_channel.*im1./(V+delta);
G_out = G_channel.*im1./(V+delta);
B_out = B_channel.*im1./(V+delta);
output = cat(3,R_out,G_out,B_out);
toc
return_img = output;
end