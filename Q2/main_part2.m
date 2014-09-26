clc; close all; clear all;

%% Reading training examples in x and y floating point arrays
fprintf('Reading Data \n');
[x1, x2] = textread('q2x.dat', '%f %f', 'delimiter', '\n', 'whitespace', '');
y = textread('q2y.dat', '%f', 'delimiter', '\n', 'whitespace', '');
y = boolean(y);

m = size(y,1);
% Number of training examples
fprintf('Number of Training examples: %d\n',m);

x = cat(2,x1,x2); % adding column of 1 to x
ext_x = cat(2,x,double(ones(m,1))); % adding column of 1 to x
n = size(x,2) + 1; 
% Number of features
fprintf('Number of features: %d\n',n);

theta = zeros(n,1); % initialising theta to zero



%% Plotting the data
figure, hold on;
plot(x(find(y==1), 1), x(find(y==1), 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(x(find(y==0), 1), x(find(y==0), 2), 'ko', 'MarkerFaceColor', 'red', 'MarkerSize', 7);

points_x = [min(x(:,1)),  max(x(:,1))];
points_y = (-1./theta(2)).*(theta(1).*points_x + theta(3));
h = plot(points_x, points_y,'LineWidth',2,'MarkerSize',10);


%% Logistic Gradient Descent Parameters - Newton's Method

iteration = 0;s = 1;

while (s>0.00001)
    iteration = iteration+1;
   
    if(iteration == 1)
        h_theta = 1./(1 + exp(-ext_x*theta));
        R_old = h_theta - y;
    else
        R_old = R_new;
    end
    
     set(h,'Xdata',points_x,'Ydata',(-1./theta(2)).*(theta(1).*points_x + theta(3)));
     drawnow; pause(0.5);
    % updating the plot every 100 loops
    % fprintf('iteratiion %d\n',i);
    % Printing number of iterations

    
    gradient = ext_x' * R_old / m;
    hessian = (repmat(1-h_theta, 1, n) .* ext_x)' * (repmat(h_theta,1,n) .* ext_x) / m;
    
    theta = theta - hessian\gradient;
    
    h_theta = 1./(1 + exp(-ext_x*theta));
    R_new = h_theta - y;
    s = abs(R_old - R_new)*10000;
    
end


hold off;


% Value of theta after Batch GDA
fprintf('Value of Theta(2nd term is the intercept term): %f,%f\n',theta(1),theta(2));
