l1 = 10; % length of first arm
l2 = 7; % length of second arm
l3 = 6;
%%traning data
theta1 = rand(1,12)*90; % all possible theta1 values
theta2 = rand(1,12)*90; % all possible theta2 values
theta3 = rand(1,12)*90;
% generate a grid of theta1 and theta2 and theta3 values
[THETA1, THETA2,THETA3] = ndgrid(theta1, theta2, theta3); 
% compute x coordinates
X = l1 * cos(THETA1*pi/180) + l2 * cos(THETA1*pi/180 + THETA2*pi/180)
 + l3*cos(THETA1*pi/180+THETA2*pi/180+THETA3*pi/180); 
  % compute y coordinates
Y = l1 * sin(THETA1*pi/180) + l2 * sin(THETA1*pi/180 + THETA2*pi/180)
 + l3*sin(THETA1*pi/180+THETA2*pi/180+THETA3*pi/180);
phi = THETA1 + THETA2 + THETA3;
% create training dataset
data = [X(:) Y(:) phi(:) THETA1(:) THETA2(:) THETA3(:)]; 

 % disorder the order
data_ = data(randperm(size(data,1)),:); %64000*6
%training data and validation(checking) data and testing data
trndata1=data_(1:round( size(data_,1)*5/7),1:4); %21600*4
trndata2=data_(1:round(size(data_,1)*5/7),[1,2,3,5]);
trndata3=data_(1:round(size(data_,1)*5/7),[1,2,3,6]); %5400*4
chkdata1=data_(round(size(data_,1)*5/7)+1:round(size(data_,1)*6/7),1:4);
chkdata2= ...
    data_(round(size(data_,1)*5/7)+1:round(size(data_,1)*6/7),[1,2,3,5]);
chkdata3= ...
    data_(round(size(data_,1)*5/7)+1:round(size(data_,1)*6/7),[1,2,3,6]);
tesdata1=data_(round(size(data_,1)*6/7)+1:size(data_,1),1:4);
tesdata2=data_(round(size(data_,1)*6/7)+1:size(data_,1),[1,2,3,5]);
tesdata3=data_(round(size(data_,1)*6/7)+1:size(data_,1),[1,2,3,6]);

%%
%theta1 predicted by anfis,traning output
fismat1=genfis2(trndata1(:,1:3),trndata1(:,4),0.25);
fprintf('-->%s\n','Start training first ANFIS network.')
[anfis1,trnErr1,ss,anfis1_,chkErr1] = ...
anfis(trndata1(:,1:4), fismat1, [250,0,.005,.9,1.1], ...
[0,0,0,0],chkdata1(:,1:4)); 
trnOut1 = evalfis(trndata1(:,1:3), anfis1); % theta1 predicted by anfis1
%trnRMSE1R=norm(trnOut1-trndata1(:,4))/sqrt(length(trnOut1));
chkOut1 = evalfis(tesdata1(:,1:3),anfis1_);
%chkRMSE1R=norm(chkOut1-tesdata1(:,4))/sqrt(length(chkOut1));
figure(1)
plot(trnErr1,'r')
hold on;
plot(chkErr1,'b')
title('Checking Error and Training Error of theta1')
xlabel('Number of Epochs')
ylabel('Angle Error(degree)')
legend('trnErrl','chkErr1')
%%
%second ANFIS network
fismat2=genfis2(trndata2(:,1:3),trndata2(:,4),0.25);
fprintf('-->%s\n','Start training second ANFIS network.')
[anfis2,trnErr2,ss,anfis2_,chkErr2] = ...
    anfis(trndata2(:,1:4), fismat2, [250,0,.005,.9,1.1], ...
    [0,0,0,0],chkdata2(:,1:4)); 
trnOut2 = evalfis(trndata2(:,1:3), anfis2); % theta1 predicted by anfis2
trnRMSE2R=norm(trnOut2-trndata2(:,4))/sqrt(length(trnOut2));
chkOut2 = evalfis(tesdata2(:,1:3),anfis2_);
%chkRMSE2R=norm(chkOut2-tesdata2(:,4))/sqrt(length(chkOut2));
figure(2)
plot(trnErr2,'r')
hold on;
plot(chkErr2,'b')
title('Checking Error and Training Error of theta2')
xlabel('Number of Epochs')
ylabel('Angle Error(degree)')
legend('trnErr2','chkErr2')
%%
%third ANFIS network
fismat3=genfis2(trndata3(:,1:3),trndata3(:,4),0.25);
fprintf('-->%s\n','Start training third ANFIS network.')
[anfis3,trnErr3,ss,anfis3_,chkErr3] = ...
    anfis(trndata3(:,1:4), fismat3, [250,0,.005,.9,1.1], ...
    [0,0,0,0],chkdata3(:,1:4)); 
trnOut3 = evalfis(trndata3(:,1:3), anfis3); % theta1 predicted by anfis3
%trnRMSE3R=norm(trnOut3-trndata3(:,4))/sqrt(length(trnOut3));
chkOut3 = evalfis(tesdata3(:,1:3),anfis3_);
%chkRMSE3R=norm(chkOut3-tesdata3(:,4))/sqrt(length(chkOut3));
figure(3)
plot(trnErr3,'r')
hold on;
plot(chkErr3,'b')
title('Checking Error and Training Error of theta3')
xlabel('Number of Epochs')
ylabel('Angle Error(degree)')
legend('trnErr3','chkErr3')
%
X2 = l1 * cos(chkOut1*pi/180) + l2 * cos(chkOut1*pi/180 + chkOut2*pi/180) 
+l3*cos(chkOut1*pi/180 + chkOut2*pi/180 + chkOut3*pi/180); 
Y2 = l1 * sin(chkOut1*pi/180) + l2 * sin(chkOut1*pi/180 + chkOut2*pi/180) 
+l3*sin(chkOut1*pi/180 + chkOut2*pi/180 + chkOut3*pi/180);
theta1_diff=tesdata1(:,4)-chkOut1;
theta2_diff=tesdata2(:,4)-chkOut2;
theta3_diff=tesdata3(:,4)-chkOut3;
x_diff=tesdata1(:,1)-X2(:);
y_diff=tesdata1(:,2)-Y2(:);
xyMSE=sum(x_diff.^2+y_diff.^2)/length(x_diff);
xyMAE=sum(abs(x_diff)+abs(y_diff))/length(x_diff);
figure()
subplot(3,1,1);
plot(theta1_diff);
ylabel('theta1 error')
title('Desired theta1 - Predicted theta1(degree)')

subplot(3,1,2);
plot(theta2_diff);
ylabel('theta2 error')
title('Desired theta2 - Predicted theta2(degree)')

subplot(3,1,3);
plot(theta3_diff);
xlabel('number of epochs')
ylabel('theta3 error')
title('Desired theta3 - Predicted theta3(degree)')

figure()
quiver(tesdata1(:,1),tesdata1(:,2),x_diff,y_diff)
%%
%RBF
net1=newrb(trndata1(:,1:3)',trndata1(:,4)',0.8,5,300,8);
net1= fitnet(8);
net1.layers{1}.transferFcn = 'radbas';
tic
net1 = train(net1,trndata1(:,1:3)',trndata1(:,4)');
toc
thetar1=sim(net1,tesdata1(:,1:3)');

net2=newrb(trndata2(:,1:3)',trndata2(:,4)',0.8,5,300,8);
net2= fitnet(8);
net2.layers{1}.transferFcn = 'radbas';
tic
net2 = train(net2,trndata2(:,1:3)',trndata2(:,4)');
toc
thetar2=sim(net2,tesdata2(:,1:3)');

net3=newrb(trndata3(:,1:3)',trndata3(:,4)',0.8,5,300,8);
net3= fitnet(8);
net3.layers{1}.transferFcn = 'radbas';
tic
net3 = train(net3,trndata3(:,1:3)',trndata3(:,4)');
toc
thetar3=sim(net3,tesdata3(:,1:3)');

Xr = l1 * cos(thetar1*pi/180) + l2 * cos(thetar1*pi/180 + thetar2*pi/180) 
+l3*cos(thetar1*pi/180 + thetar2*pi/180 + thetar3*pi/180); 
Yr = l1 * sin(thetar1*pi/180) + l2 * sin(thetar1*pi/180 + thetar2*pi/180) 
+l3*sin(thetar1*pi/180 + thetar2*pi/180 + thetar3*pi/180);
thetar1_diff=tesdata1(:,4)-thetar1(:);
thetar2_diff=tesdata2(:,4)-thetar2(:);
thetar3_diff=tesdata3(:,4)-thetar3(:);
xr_diff=tesdata1(:,1)-Xr(:);
yr_diff=tesdata1(:,2)-Yr(:);
%error=sum(sqrt(xr_diff.^2+yr_diff.^2))/length(xr_diff);
xyrMSE=sum(xr_diff.^2+yr_diff.^2)/length(xr_diff);
xyrMAE=sum(abs(xr_diff)+abs(yr_diff))/length(xr_diff);
figure()
subplot(3,1,1);
plot(thetar1_diff);
ylabel('theta1 error')
title('Desired theta1 - Predicted theta1(degree)')

subplot(3,1,2);
plot(thetar2_diff);
ylabel('theta2 error')
title('Desired theta2 - Predicted theta2(degree)')

subplot(3,1,3);
plot(thetar3_diff);
ylabel('theta3 error')
title('Desired theta3 - Predicted theta3(degree)')

figure()
quiver(tesdata1(:,1),tesdata1(:,2),xr_diff,yr_diff)