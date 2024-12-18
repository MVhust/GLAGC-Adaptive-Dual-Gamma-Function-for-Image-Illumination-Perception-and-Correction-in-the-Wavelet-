%% GSLF
function param= GSLF_function(cA)
hist_cA = imhist(mat2gray(cA));
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