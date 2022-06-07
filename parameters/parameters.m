clear
close all
% load RCNN labeled data data
train_x2 = load('carorigimgs.txt');
train_x = train_x2(:,1:end-1)';
train_y = train_x2(:,end);
T = transformtarget(train_y+1,2)';

% rnn matrix
[n1 n2] = size(train_x);
n3 = floor(0.5*n2);
n4 = floor(0.8*n2);
X = train_x;
neu = 3;
net = layrecnet(1:2,neu);
net.trainFcn = 'trainbr';
net.trainParam.epochs = 1;
net.trainParam.showWindow = false;
net.trainParam.showWindow=0;
net.divideFcn = 'divideind'; 
net.performFcn = 'mse'; 
net.divideParam.trainInd = 1:n3;
net.divideParam.valInd   = n3+1:n4;
net.divideParam.testInd  = n4+1:size(train_x,2);

for a1 = 1:10%size(a,2)
[net,tr] = train(net,X,T);
W0 = net.LW{1,1}(:,1:neu);
W1 = net.LW{1,1}(:,neu+1:2*neu);
K = net.LW{2,1};

Y = net(X);
errnn(a1) = mse(Y,T);
if a1 == 1
    error0 = errnn(a1);
end

% compute alpha and beta for sdp
%lambda_min and lambda_max of input layer weights
n = neu;
A = cov(T');
A = diag(A(1:n));
lambda = eig(A);
lambda_min = min(lambda);
lambda_max = max(lambda);

% Q, epsilon is a constant > 0, d is dist delay < 1, R is interconnenction matrix
d = 0.4; %rand(1)
R = 4*eye(n,n);

% inequality, h is max delay
h = 2;
beta = sdpvar(1,1,'full');
P = sdpvar(n,n,'full'); % for Lyapunov
epsilon1 = sdpvar(1,1,'full');
epsilon2 = sdpvar(1,1,'full');
epsilon3 = sdpvar(1,1,'full');

Q = (epsilon2*(1-d)^-1)*(R')*R;
K = sdpvar(n,2,'full'); % gain matrix
F = 4*eye(n);
Pi1 = (epsilon1^-1)*W0*(W0')+(epsilon2^-1)*W1*(W1')+(epsilon3^-1)*K*(K');
Pi = (A')*P + P*A + P*Pi1*P+epsilon1*(R')*R+Q + epsilon2*(F')*F;

alpha1 = min(eig(P))^-1;
alpha = max(eig(P))+h*max(eig(Q))*(1+beta{1}*h*exp(beta{1}*h))*alpha1{1};

error_g = alpha*exp(-1*beta{1}*h);

O = [P>0,beta>0,error_g<error0,min(eig(-Pi))-beta{1}*max(eig(P))-beta{1}*h*max(eig(Q))*exp(beta{1}*h)>=0,epsilon1>=0,epsilon2>=0,epsilon3>=0];
solvesdp(O); 
checkset(O);

%save parap.mat alpha beta h error0
betaarr(a1) = double(beta);
alpharr(a1) = double(alpha);

net.LW{2,1} = double(K)';

close all

end

plot(errnn)