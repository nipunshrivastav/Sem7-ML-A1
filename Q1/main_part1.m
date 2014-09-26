 clc; close all; clear all;

%% Reading training examples in x and y floating point arrays
fprintf('Reading Data \n');
x = textread('q1x.dat', '%f', 'delimiter', '\n', 'whitespace', '');
y = textread('q1y.dat', '%f', 'delimiter', '\n', 'whitespace', '');

x = x-mean(x);
x = x./std(x);

m = size(x,1);
% Number of training examples
fprintf('Number of Training examples: %d\n',m);
n = size(x,2) + 1;
% Number of features
fprintf('Number of features: %d\n',n);


%% Gradient Descent Parameters

eta = 0.1; % Learning Rate
fprintf('Value of Learning Rate: %f\n',eta);
eta = eta/m;


ext_x = cat(2,x,double(ones(m,1))); % adding column of 1 to x

theta = [0; 0]; % initialising theta to zero



%% Gradient Descent

figure;
scatter(x,y);
hold on
h=plot(x,y,'LineWidth',2,'MarkerSize',10,'color','red');
% Plot showing learnt hypothesis currently
% gets updated in each iteration of gradient descent



s = 2;counter=0;
% initialising s and loop counter
threshold = 0.0005;
fprintf('Stopping Criterion: Change is J_theta(x) is less than %f\n',threshold);
while (s>threshold)
    
    if(mod(counter,1)==0)
         set(h,'Xdata',x,'Ydata',ext_x*theta);
         drawnow;
      %   pause(0.2);
    end
    % updating the plot every 1000 loops

    counter = counter+1; % Loop Count
    
    if(counter == 1)
        s = 10;
    else
        R_old = R_new;
    end
    % Updating R_old with the R_new of last loop
    % Special Condition for first loop
        
    
    h_theta = ext_x*theta;
    error = y - h_theta;
    % value of differentiation of J_theta without multiplied with x
    
    theta(2) = theta(2) + eta*sum(error);
    theta(1) = theta(1) + eta*sum(error'*x);
    % updating thetas
    
    R_new = sum(error.*error);
    % updating R_new, which is J_theta with new theta obtained
    
    
    if(counter>1)
        s = abs(R_old - R_new);
    end
    % seeing the change in J_theta
    allJ(counter) = R_new./2;
    alltheta(counter,:)=theta;
    
end
 set(h,'Xdata',x,'Ydata',ext_x*theta);
 drawnow;
 pause(0.2);
 hold off;

 
 counter
% Value of theta after Batch GDA
fprintf('Value of Theta(2nd term is the intercept term): %f,%f\n',theta(1),theta(2));


%% 3-dimensional mesh showing the error function

range = 3; interval = 0.1;
% Initialising parameters for surface plot

[theta1a,theta2a] = meshgrid(theta(1)-range:interval:theta(1)+range,theta(2)-range:interval:theta(2)+range);
% making a mesh of theta1 and theta2 in order to make a surface plot
% around the obtained theta1 and theta2

Z = 0;
for i=1:m
    Z = Z + 0.5*((theta1a*(x(i))+theta2a)-y(i)).^2;
end
% calculating j_theta for every combination of theta1 and theta2


figure
meshc(theta1a,theta2a,Z);
alpha = 0;
beta = 0;
gamma = 0;
hold on;
P = plot3(alpha,beta,gamma,'ro','MarkerSize', 10, 'LineWidth', 5);
P1 = plot3(alpha,beta,0,'ro','MarkerSize', 10, 'LineWidth', 5);
for k = 1:counter
    set(P,'XData',alltheta(k,1),'YData',alltheta(k,2),'ZData',allJ(k));
    set(P1,'XData',alltheta(k,1),'YData',alltheta(k,2),'ZData',0);
    view(45,5+20/k);
    drawnow;
    pause(0.2);
end

set(P,'XData',alltheta(k,1),'YData',alltheta(k,2),'ZData',allJ(k));
drawnow;
pause(0.2);

hold off;


% Contour plot
figure, contour(theta1a, theta2a, Z,50);
hold on;
h = plot(theta(1), theta(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
% plotting the point in the contour plot

for i = 1:counter
    set(h,'Xdata',alltheta(i,1),'Ydata',alltheta(i,2));
    drawnow;
    pause(0.2);
end

set(h,'Xdata',alltheta(i,1),'Ydata',alltheta(i,2));
drawnow;
pause(0.2);

hold off;