% Clear workspace
clear
close all
clc;

figure 
hold on
view(30,30);
%showRef();
H = quad2H([[242.5,450.5];[24.5,727];[1894,708];[1629.5,445]],[[0.0,0.0];[15,0];[15,28];[0,28]]);
p = [1.177e+03;6.020e+02;1];
p_W = H*p;
p_W(1) = p_W(1)/p_W(3);
p_W(2) = p_W(2)/p_W(3);
figure 
hold on
poly = polyshape([242,1629,1894,24],[450,445,708,727]);
plot(poly)
%plot(p(1),p(2),'o','markersize',5,'markerfacecolor','red');
figure 
hold on 
poly = polyshape([0,15,15,0],[0,0,28,28]);
plot(poly,'facecolor','red')
keyboard;
homo()
keyboard;
A0 = [242,450];
S8 = [1894,708];
S0 = [1629,445];
A8 = [24,727];
p = polyshape([242,1629,1894,24],[450,445,708,727]);
box = polyshape([240,276,276,240],[359,359,453,453]);
figure
hold on
%plot(box)
plot(p)

T = readtable('data/court_geometry.xlsx');
l = ['A','J','S'];
for i = 1:length(l)
p0_str = strcat(l(i),'0');
p1_str = strcat(l(i),'8');
p0 = T{ismember(T{:,1},p0_str),7:end};
p1 = T{ismember(T{:,1},p1_str),7:end};
drawline(p0,p1,4);
end
l = ['A':'F','N':'S'];
for i = 1:length(l)
p0_str = strcat(l(i),'2');
p1_str = strcat(l(i),'6');
p0 = T{ismember(T{:,1},p0_str),7:end};
p1 = T{ismember(T{:,1},p1_str),7:end};
drawline(p0,p1,12);
end
l = ['G','H','I','J','K','L','M','Q'];
for i = 1:length(l)
p1_str = strcat(l(i),'4');
p0 = T{ismember(T{:,1},'C4'),7:end};
plot(p0(1),p0(2),'o','markersize',5,'markerfacecolor','blue');
p1 = T{ismember(T{:,1},p1_str),7:end};
plot(p1(1),p1(2),'o','markersize',5,'markerfacecolor','blue');
%drawline(p0,p1,12);
end
A0 = T{ismember(T{:,1},'A0'),7:end};
J0 = T{ismember(T{:,1},'J0'),7:end};
C4 = T{ismember(T{:,1},'K4'),7:end};
H4 = T{ismember(T{:,1},'H4'),7:end};
A8 = T{ismember(T{:,1},'A8'),7:end};
J8 = T{ismember(T{:,1},'J8'),7:end};

T1 = T{ismember(T{:,1},'T1'),7:end};
T2 = T{ismember(T{:,1},'T2'),7:end};
T3 = T{ismember(T{:,1},'T3'),7:end};
T4 = T{ismember(T{:,1},'T4'),7:end};
drawline(T1,T4,20);
drawline(T2,T3,20);
exportgraphics(gca,strcat('court1','.jpeg'))
%drawline(A0,J0,2);
%drawline(C4,H4,15);
%drawline(A8,J8,2);
%axis equal

keyboard;
c1 = [99  96  81]/255;
c2 = [75  81 108]/255;
c3 = [119 121 148]/255;

c1_lab = rgb2lab(c1)
c2_lab = rgb2lab(c2)
c3_lab = rgb2lab(c3)

teste = rgb2lab([111.8423577793009, 108.69910897875258, 141.85058259081563]/255);
norm(teste - c1_lab)
norm(teste - c2_lab)
norm(teste - c3_lab)
keyboard


function drawline(p1,p2,fac)
L = norm(p2 - p1);
d = (p2 - p1)/L;
margin = fac*L;
pi = p1 - margin*d;
pj = p2 + margin*d;
line([pi(1),pj(1)],[pi(2),pj(2)]);
end
function homo(p0,p1,p2,p3)
figure 
hold on
view(30,30)
p = [0,0,0];
p0 = [1,0,0];
p1 = [0,1,0];
p2 = [0,0,1];
p3 = [1,1,1];
n = cross(p1-p0,p2-p0);
n = n/norm(n);
line([p0(1),p1(1)],[p0(2),p1(2)],[p0(3),p0(3)]);
line([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)]);
line([p2(1),p0(1)],[p2(2),p0(2)],[p2(3),p0(3)]);
line([p(1),p3(1)],[p(2),p3(2)],[p(3),p3(3)]);
p3 = p3/norm(p3);
plot3(p0(1),p0(2),p0(3),'o','markersize',7,'markerfacecolor','red');
plot3(p1(1),p1(2),p1(3),'o','markersize',7,'markerfacecolor','red');
plot3(p2(1),p2(2),p2(3),'o','markersize',7,'markerfacecolor','red');
plot3(p3(1),p3(2),p3(3),'o','markersize',7,'markerfacecolor','red');

p0 = [1,-1,1];
p1 = [1,-2,2];
p2 = [-1,2,-1];
p3 = [0,1,2];
line([p0(1),p1(1)],[p0(2),p1(2)],[p0(3),p1(3)]);
line([p1(1),p3(1)],[p1(2),p3(2)],[p1(3),p3(3)]);
line([p3(1),p2(1)],[p3(2),p2(2)],[p3(3),p2(3)]);
line([p2(1),p0(1)],[p2(2),p0(2)],[p2(3),p0(3)]);

plot3(p0(1),p0(2),p0(3),'o','markersize',7,'markerfacecolor','blue');
text(p0(1),p0(2),p0(3),'p0');
plot3(p1(1),p1(2),p1(3),'o','markersize',7,'markerfacecolor','blue');
text(p1(1),p1(2),p1(3),'p1');
plot3(p2(1),p2(2),p2(3),'o','markersize',7,'markerfacecolor','blue');
text(p2(1),p2(2),p2(3),'p2');
plot3(p3(1),p3(2),p3(3),'o','markersize',7,'markerfacecolor','blue');
text(p3(1),p3(2),p3(3),'p3');
end
function M = quad2H(p_1,p_2)
n = [1,1,1];
z = n/norm(n);
y = cross(z,-[1,0,0]);
y = y/norm(y);
x = cross(y,z);
%quiver3(n(1),n(2),n(3),n(1)+x(1),n(2)+x(2),n(3)+x(3));
%quiver3(n(1),n(2),n(3),n(1)+y(1),n(2)+y(2),n(3)+y(3));
T = [x',y',n'];
p_1(:,3) = 1.0;
p_2(:,3) = 1.0;
H_1 = buildH(T*p_1');
H_2 = buildH(T*p_2');
M = inv(T)*H_2*inv(H_1)*T;


end
function showRef()
p = [0,0,0];
p0 = [1,0,0];
p1 = [0,1,0];
p2 = [0,0,1];
p3 = [1,1,1];
n = cross(p1-p0,p2-p0);
n = n/norm(n);
line([p0(1),p1(1)],[p0(2),p1(2)],[p0(3),p0(3)]);
line([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)]);
line([p2(1),p0(1)],[p2(2),p0(2)],[p2(3),p0(3)]);
line([p(1),p3(1)],[p(2),p3(2)],[p(3),p3(3)]);
p3 = p3/norm(p3);
plot3(p0(1),p0(2),p0(3),'o','markersize',7,'markerfacecolor','red');
plot3(p1(1),p1(2),p1(3),'o','markersize',7,'markerfacecolor','red');
plot3(p2(1),p2(2),p2(3),'o','markersize',7,'markerfacecolor','red');
plot3(p3(1),p3(2),p3(3),'o','markersize',7,'markerfacecolor','red');
end
function H = buildH(p)
p0 = p(:,1);
p1 = p(:,2);
p2 = p(:,3);
p3 = p(:,4);
H = [p0,p1,p2];
lam = H\p3;
H = [lam(1)*p0,lam(2)*p1,lam(3)*p2];
end







