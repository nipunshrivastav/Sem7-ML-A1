clc;  close all; clear all;

%% Reading training examples in x and y floating point arrays
fprintf('Reading Data \n');
x = textread('q3x.dat', '%f', 'delimiter', '\n', 'whitespace', '');
y = textread('q3y.dat', '%f', 'delimiter', '\n', 'whitespace', '');


m = size(x,1);
% Number of training examples
fprintf('Number of Training examples: %d\n',m);
n = size(x,2) + 1;
% Number of features
fprintf('Number of features: %d\n',n);
ext_x = cat(2,x,double(ones(m,1))); % adding column of 1 to x

%% Locally Weighted Linear Regression

W = eye(m);% Weight matrix
alltheta = zeros(m,n);% Will containt thetas corresponding to 
tau = 0.8;

for j = 1:m % Runs by taking points from data set as a pseudo query point and accordingly distributing weights
    
    for i = 1:m
        W(i,i) = exp(-(x(i)-x(j))^2/(2*(tau^2)));% ASK WHETHER ONE NUMBER OR WHOLE ARRAY
    end
    % Assigning weight matrix

    theta = zeros(n,1);% initialising theta

    %J_theta = ((ext_x*theta - y)'*W*(ext_x*theta - y))/2;
    %del_J_theta = (ext_x)'*W*ext_x*theta - ext_x'*W*y;

    theta = ((ext_x)'* W* ext_x)\((ext_x)'* W* y);
    alltheta(j,:) = theta';
end
    thetaW = (ext_x'*ext_x)\ext_x'*y;

    %% Plotting the weighted and unweighted function

    figure;
    scatter(x,y);
    hold on
    
    z = zeros(m,1);
    
    for j = 1:m
        theta = [alltheta(j,1);alltheta(j,2)];
        temp = ext_x*theta;
        z(j) = temp(j);    
    end
  [Xsorted, SortIndex] = sort(x);
  Zsorted = z(SortIndex);

   plot(Xsorted,Zsorted, 'LineWidth', 2.5, 'color', 'red');
   plot(x,ext_x*thetaW,'LineWidth',2,'MarkerSize',10);
    % Plot showing weighted and unweighted cases
    hold off;
   
fprintf('Value of Theta for unweighted case: %f,%f\n',thetaW(1),thetaW(2));
