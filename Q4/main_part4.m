clc; close all; clear all;

%% Reading training examples in x and y floating point arrays
fprintf('Reading Data \n');
[x1, x2] = textread('q4x.dat', '%f %f', 'delimiter', '\n', 'whitespace', '');
Y = textread('q4y.dat', '%s', 'delimiter', '\n', 'whitespace', '');

y(strcmp('Alaska',Y)) = 1;
y(strcmp('Canada',Y)) = 0;
y = y'; % Map Alaska to 1 and other to 0

m = size(y,1);
% Number of training examples
fprintf('Number of Training examples: %d\n',m);

x = cat(2,x1,x2); % adding column of 1 to x
n = size(x,2); 
% Number of features
fprintf('Number of features: %d\n',n);

theta = zeros(n,1); % initialising theta to zero


%% Discriminant Analysis

k = 0;j = 0; % K and J hold the number of cases where y = 1 and y = 0 respectively
for i = 1:m
    if (y(i) == 1)
        k = k + 1;
        x1A(k,:) = x(i,:);
    else
        j = j + 1; 
        x0A(j,:) = x(i,:);
    end
end
% x1A has all x for which y = 1 and similarly for y = 0 we have x0A

fprintf('Value of mean of x for y == Alaska:\n');
mu1 = mean(x1A)
fprintf('Value of mean of x for y == Canada:\n');
mu2 = mean(x0A)



%% Covariance
coVar = zeros(n,n);

x1A = (x1A-repmat(mu1,k,1));
x0A = (x0A-repmat(mu2,k,1));
% Subtracting each term with mean

for i = 1:k
   cur = (x1A(i,:)')*(x1A(i,:));
   coVar = coVar + cur;
end

S1 = coVar;

for i = 1:j
   cur = (x0A(i,:)')*(x0A(i,:));
   coVar = coVar + cur;
end

S2 = (coVar - S1)./j;
S1 = S1./k;

fprintf('Value of Covariance Matrix:\n');
coVar = coVar./m


%% LDA and QDA

 r = (((coVar)\mu1')-((coVar)\mu2'))+((mu1/coVar)-(mu2/coVar))';
 p = r(1);q = r(2);
 
 
R1 = min(x(:,1)):0.1:max(x(:,1));
R2 = -R1*(p/q)+((mu1*(((coVar)\mu1')))-(mu2*((coVar)\mu2')))/q;


linear =  (S1\mu1'-S2\mu2')+(mu1/S1-mu2/S2')';
constant =  log(1/det(S1)^2)-log(1/det(S2)^2)+((mu1*((S1)\mu1'))-(mu2*((S2)\mu2')));

final = @(x,y) ([x y]*(inv(S2)-inv(S1))*([x y]'))+([x y]* linear)-constant;



%% Plotting the data
figure, hold on;
plot(x(find(y==1), 1), x(find(y==1), 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(x(find(y==0), 1), x(find(y==0), 2), 'ko', 'MarkerFaceColor', 'red', 'MarkerSize', 7);
hold on;
plot(R1,R2,'LineWidth',2,'MarkerSize',10);
% Linear Discriminant Analysis Plot

h = ezplot(final,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
% Quadratic Discriminant Analysis Plot
set(h,'LineWidth',2,'Color','green');
hold off;


