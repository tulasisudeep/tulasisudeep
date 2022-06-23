

Experiment-1
Basics of signal processing
1. Plot the following signal
The first section in the figure is an exponentially decaying signal 𝑒
−0.5𝑡
.
2. For the signal x(t) given below 
𝑥(𝑡) = 2𝑡 ,−1 < 𝑡 ≤ 0
= 𝑒
−𝑡 + sin (2𝜋𝑡) ,0 < 𝑡 < 5
a) Plot 𝑥(𝑡)
b) Plot 𝑥(
𝑡
2
+ 0.5)
c) Compute even component of 𝑥(𝑡) and plot
3. Find the convolution and plot it for the given signal, 𝑥(𝑡) = 𝑒
−𝑡
and
 ℎ(𝑡) = 𝛿(𝑡 + 2) − (𝑡 − 2).
3
Experiment 2
Sampling of analog signals and study of aliasing
Aim: To verify sampling theorem for Nyquist rate, Under sampling and Oversampling 
condition
1. A signal 𝑥(𝑡) = cos (2𝜋𝑓𝑚𝑡) with fm=0.02 is sampled at frequencies
a) fs1 = 0.02
b) fs2 = 0.04
c) fs3 = 0.5
Generate the signal x(t) and the discrete time signals x[n] for the corresponding sampling 
frequencies.
2. Consider an input signal 𝑥(𝑡) = cos(0.04𝜋𝑡) + sin (0.08𝜋𝑡) applied as an input to an 
Analog to Digital (A/D) converter shown below:
Find the discrete time signal obtained at the output of the A/D converter if the sampling 
rate of the A/D converter is a) fs1=0.04Hz b) fs2=0.08Hz c) fs3=0.5Hz
Generate the signal x(t) and the discrete time signals x[n] for the corresponding sampling 
frequencies. What do you observe?
Experiment 3
Computation of Discrete Fourier Transform (DFT) using Direct/Linear Transformation 
Method
Aim: To study the computation of DFT using linear transformation in MATLAB.
1. A discrete time signal is given as 𝑥(𝑛0 = 1 𝑓𝑜𝑟 0 ≤ 𝑛 ≤ 7.
Find the DFT of x(n) for N=8, N=16 and N=64. (Use MATLAB command fft). Plot 
magnitude response and phase spectrum.
2. Consider a sinusoidal signal x(t) of amplitude 1V and frequency 5Hz. Plot magnitude and 
phase spectrum of x(t) using DFT.
3. Consider a sinusoidal signal x1(t) which is sum of three sinusoidal signals of amplitude 1V 
each and frequencies 5Hz, 10Hz and 20Hz. Plot the magnitude and phase spectrum of x(t).
Analog to Digital (A/D) Converter
x(t) y(t)
4
4. Consider a sinusoidal signal x2(t) which is concatenation of three sinusoidal signals of 
amplitude 1V each and frequencies 5Hz, 10Hz and 20Hz. Plot the magnitude and phase 
spectrum of x(t). What do you infer?
Experiment 4
Properties of DFT
Aim : To verify the different properties of DFT for the given signals using MATLAB.
 1 Perform circular folding on a discrete time sequence x[n] of length N. 
 Consider x[n] = [1 2 3 -2 6 4] and N=8.
2. Perform circular time shift of ‘m’ samples on a discrete time sequence x[n] of length N for the 
following cases:
a) m is positive integer
b) m is negative integer
 Consider x[n] = [1 3 -2 0 5 7 2 -1] and m=3.
3. Find and plot even and odd components of a sequence x[n] 
 Consider x[n] = [1 -4 2 6 -3 7 4 1].
4. Find DFT X(k) of a sequence x[n]. Use X(k) and suitable property of DFT to find DFT of 
x((n-3)10).
5. Find DFT X(k) of a sequence x[n]. Use X(k) and suitable property of DFT to find DFT of x((-
n)16). Consider x[n] = [1 2 3 -7 6 4 5 9].
6. Find the circular convolution of the signals x[n] = [5 4 3 2 2 4 6 6 8] and h[n] = [4 3 2 1] using 
DFT computation. Determine the linear convolution of the above-mentioned sequences and 
compare it with circular convolved output.
7. Use Parseval’s theorem to find the energy of the signal x[n] = [1 5 3 7 -20 12 8 2].
8. Verify that the DFT of a real and odd sequence is imaginary and odd using suitable example. 
Take N=8.
Experiment 5
Computation of 2N-point DFT of a real sequence using N- point DFT
Aim: To compute a 2-N point DFT of a real sequence by using a N point DFT only once.
5
1. Find the 8 point DFT of the signals x[n] = [5 4 3 2 2 4 6 8] and h[n] = [4 3 2 1] by 
computing DFT only once.
2. Find 16 point DFT of the real sequence g[n] by computing 8 point DFT on the complex 
sequence x[n]. It is given that g[n] = n, 3≤ 𝑛 ≤ 11 and g[n] = 0 for 0≤ 𝑛 < 3 and 11<
𝑛 ≤ 15 and x[n] = g[2n] +j g[2n+1].
Experiment 6
Linear filtering using overlap add/save method
Aim : To perform block convolution on the given sequence by using overlap add/ save 
method in MATLAB.
1. Compute the block convolution using overlap save method for the input sequence x[n] = 
[ 1 2 -1 2 3 -2 -3 -1 1 1 2 -1] and the impulse response h[n] = [ 1 2 3 -1].
2. Perform block convolution for the input sequence given in Qn.1 using overlap add 
method and verify the result.
.
Experiment 7
Design of Infinite Impulse Response ( IIR) Filter
Aim: To design a IIR Butterworth and Chebyshev type 1 filter according to the given 
specifications and to plot the frequency response of the filter using MATLAB.
1. Design a Digital Butterworth IIR low pass filter for the following specifications
Passband edge frequency 400Hz
Stopband edge frequency 800Hz
Sampling frequency 2000Hz
Stopband attenuation 30dB
Passband attenuation 0.4dB 
Plot the frequency response of the filter. Also design high pass, band pass and band stop 
filters with appropriate specifications.
2. Design a Digital IIR Chebyshev type 1 band pass filter for the given specifications:
Passband attenuation of 2dB at passband edge frequencies 0.2π and 0.4π.
Stopband attenuation of 20dB at stopband edge frequencies 0.1π and 0.5π
Experiment 8
Design of Finite Impulse Response (FIR) filter
AIM: To design FIR filters for the given specifications for all the different windowing 
6
techniques using MATLAB.
Design the following FIR filter for the given specifications:
1. Low pass filter with cut off frequency 0.6π, order N=21 using Rectangular Window and 
plot the frequency response of the filter. Repeat part (1) using following window 
functions and compare the frequency responses
a) Hamming Window
b) Blackman Window
c) Hanning Window
d) Bartlett Window
2. High pass filter with cut off frequency 0.3π, order N=21 using Rectangular Window and 
plot the frequency response of the filter. Repeat part (2) using following window 
functions and compare the frequency responses
a) Hamming Window
b) Blackman Window
c) Hanning Window
d) Bartlett Window
3. Band pass filter with cut off frequencies 0.3π and 0.8π, order N=21 using Rectangular 
Window and plot the frequency response of the filter. Repeat part (2) using following 
window functions and compare the frequency responses
a) Hamming Window
b) Blackman Window
c) Hanning Window
d) Bartlett Window
4. Band stop filter with cut off frequencies 0.4π and 0.7π, order N=21 using Rectangular 
Window and plot the frequency response of the filter. Repeat part (2) using following 
window functions and compare the frequency responses
a) Hamming Window
b) Blackman Window
c) Hanning Window
d) Bartlett Window
Worksheet 9
Applications of DSP: A few case studies
Aim: To study the applications of DSP by designing Butterworth and Chebyshev type-1 IIR 
filter according to the given conditions in MATLAB.

1. Design a 4th order digital Chebyshev type 1 filter that passes only the component 800πt 
from the digital x(t) is given by 
x(t) = 2sin(100πt) + 3cos(800πt) + sin(1600πt)
Sketch the input signal, the filter’s magnitude response and the signal’s spectra before and 
after filtering.
2. Design a 6th order Butterworth filter to remove sin(80πt) component from x(t)=2+sin(80πt) 
+sin(200πt). Sketch the input signal, the filter’s magnitude response and the signal’s spectra 
before and after filtering.




experiment 1
1)clc;
clear all;
close all;
t= -3:0.001:4;
x1= (exp(-0.5*t).*(t> -3 & t<=-1));
x2= (-3*t+1).*(t> -1 & t<=0);
x3= (1).*(t>0 & t<=2);
T=0.4;
w= 2*pi/T;
x4= (exp(0.5*t).*(sin(w*t)));
x=x1+x2+x3+(x4.*(t>=2 & t<4));
plot(t,x)
xlabel('t');
ylabel('amplitude');
legend('x(t)');
grid;

2)clc;
clear;
close all;
t=-5:0.01:5;
x1=exp(-t).*(t>=-5 & t<=5);
x2=((1).*(t == -2))+((-1).*(t == 2));
subplot(3,1,1), plot(t,x1);
xlabel('t');
ylabel('amplitude');
grid;
legend('x1');
subplot(3,1,2), plot(t,x2);
xlabel('t');
ylabel('amplitude');
grid;
legend('x2');
y=conv(x1,x2);
t1= -10:0.01:10;
subplot(3,1,3), plot(t1,y);
xlabel('t');
ylabel('amplitude');
grid;
legend('y(t)');


3)function[x]=expfun(t)
x1=2*t;
x2=exp(-1*t)+sin(2*3.14*t);
x=(x1.*(t>-1 & t<=0))+ (x2.*(t>0 & t<5));


clc;
clear all;
close all;
t=-5:0.001:5;
a=expfun(t);
subplot(2,2,1),plot(t,a);
xlabel('t');
ylabel('amplitude');
grid;
legend('a');
b=expfun((t/2)+0.5);
subplot(2,2,2),plot(t,b);
xlabel('t');
ylabel('amplitude');
grid;
legend('b');
c=((1/2).*(expfun(t)+expfun(-t)));
subplot(2,2,3),plot(t,c);
xlabel('t');
ylabel('amplitude');
grid;
legend('c');
d=((1/2).*(expfun(t)-expfun(-t)));
subplot(2,2,4),plot(t,d);
xlabel('t');
ylabel('amplitude');
grid;
legend('d');


experiment 2
1)clc;
clear all;
close all;
t=-100:0.001:100;
fm=0.02;
x=cos(2*pi*fm*t);
subplot(2,2,1),plot(t,x);
xlabel('t');
ylabel('amplitude');
legend('x(t)')
grid;
fs1=fm; 
fs2=(2*fm); 
fs3=(10*fm); 
n1= (-100*fs1):1:(100*fs1);
n2=(-100*fs2):1:(100*fs2);
n3=(-100*fs3):1:(100*fs3);
x1=cos((2*pi*fm*n1)/fs1);
subplot(2,2,2),stem(n1,x1);
xlabel('t');
ylabel('amplitude');
legend('x(1)')
grid;
x2=cos((2*pi*fm*n2)/fs2);
subplot(2,2,3),stem(n2,x2);
xlabel('t');
ylabel('amplitude');
legend('x(2)')
grid;
x3=cos((2*pi*fm*n3)/fs3);
subplot(2,2,4),stem(n3,x3);
xlabel('t');
ylabel('amplitude');
legend('x(3)')
grid;


2)clc;
clear;
close all;
t=-100:0.001:100;
x = cos(0.04*pi*t)+sin(0.08*pi*t);
subplot(2,2,1);
grid;
plot(t,x);
xlabel('t');
ylabel('amplitude');
legend('continous signal');
fs1=0.04;
fs2=0.08;
fs3=0.5;
n1= -100*fs1:1:100*fs1;
n2= -100*fs2:1:100*fs2;
n3= -100*fs3:1:100*fs3;
x1= cos(0.04*pi*n1/fs1)+sin(0.08*pi*n1/fs1);
subplot(2,2,2);
stem(n1,x1);
grid;
hold;
legend('undersampling')
xlabel('n1');
ylabel('x1');
x2= cos(0.04*pi*n2/fs2)+sin(0.08*pi*n2/fs2);
subplot(2,2,3);
grid;
stem(n2,x2);
hold;
legend('nyquist')
xlabel('n2');
ylabel('x2');
x3= cos(0.04*pi*n3/fs3)+sin(0.08*pi*n3/fs3);
subplot(2,2,4);
grid;
stem(n3,x3);
hold;
legend('oversampling')
xlabel('n3');
ylabel('x3');


experiment 3 
1)clc;
clear all;
close all;
n=0:1:7;
x=[1,1,1,1,1,1,1,1];
X=fft(x,8);
subplot(3,2,1)
stem(n,X)
xlabel('n');
ylabel('magnitude');
legend('MR of 8 point dft');
grid;
subplot(3,2,2)
stem(angle(X));
xlabel('n');
ylabel('phase');
legend('Phase of 8 point dft');
grid;
Y=fft(x,16);
k=0:16-1;
subplot(3,2,3)
stem(k,Y);
xlabel('n');
ylabel('magnitude');
legend('MR of 16 point dft');
grid;
subplot(3,2,4)
stem(angle(Y));
xlabel('n');
ylabel('phase');
legend('phase of 16 point dft');
grid;
Z=fft(x,64);
k=0:64-1;
subplot(3,2,5)
stem(k,Z);
xlabel('n');
ylabel('magnitude');
legend('MR of 64 point dft');
grid;
subplot(3,2,6)
stem(angle(Z));
xlabel('n');
ylabel('phase');
legend('phase of 64 point dft');



2) clc;
clear all;
close all;
A=1;
f=5;
fs=50;
ts=(1/fs);
w=2*pi*f;
n=0:1:fs;
N=10;
x=sin(w*n*ts);
X=fft(x,8);
K1=0:length(X)-1;
subplot(3,2,1);
stem(K1,abs(X));
grid;
xlabel('len');
ylabel('x[n]');
subplot(3,2,2);
stem(K1,angle(X));
grid;
xlabel('len');
ylabel('x[n]');
Y=fft(x,16);
K2=0:length(Y)-1;
subplot(3,2,3);
stem(K2,abs(Y));
grid;
xlabel('len');
ylabel('x[n]');
subplot(3,2,4);
stem(K2,angle(Y));
grid;
xlabel('len');
ylabel('x[n]');
Z=fft(x,64);
K3=0:length(Z)-1;
subplot(3,2,5);
stem(K3,abs(Z));
grid;
xlabel('len');
ylabel('x[n]');
subplot(3,2,6);
stem(K3,angle(Z));
grid;xlabel('len');
ylabel('x[n]');


3) clc;
clear all;
close all;
A=1;
f1=5;
f2=10;
f3=20;
fs=200;
ts=(1/fs);
w1=2*pi*f1;
w2=2*pi*f2;
w3=2*pi*f3;
n=0:1:fs;
x=cos(w1*n*ts)+cos(w2*n*ts)+cos(w3*n*ts);
X1=fft(x,8);
K1=0:length(X1)-1;
subplot(3,2,1);
stem(K1,abs(X1));
grid;
xlabel('length');
ylabel('x(t)');
subplot(3,2,2);
stem(K1,angle(X1));
grid;
xlabel('length');
ylabel('x[n]');
X2=fft(x,16);
K2=0:length(X2)-1;
subplot(3,2,3);
stem(K2,abs(X2));
grid;
xlabel('length');
ylabel('x(t)');
subplot(3,2,4);
stem(K2,angle(X2));
grid;
xlabel('length');
ylabel('x[n]');
X3=fft(x,64);
K3=0:length(X3)-1;
subplot(3,2,5);
stem(K3,abs(X3));
grid;
xlabel('length');
ylabel('x(t)');
subplot(3,2,6);
stem(K3,angle(X3));
grid;
xlabel('length');
ylabel('x[n]');


4) clc;
clear;
close;
A=1;
f1=5;f2=10;f3=20;
fs=200;
ts=(1/fs);
w1=2*pi*f1;w2=2*pi*f2;w3=2*pi*f3;
n=0:1:fs;
x1=cos(w1*n*ts);
x2=cos(w2*n*ts);
x3=cos(w3*n*ts);
x=[x1 x2 x3];
X1=fft(x,8);
K1=0:length(X1)-1;
subplot(3,2,1);
stem(K1,abs(X1));
grid;xlabel('length');
ylabel('x[n]');
subplot(3,2,2);
stem(K1,angle(X1));
grid;
xlabel('length');
ylabel('x[n]');
X2=fft(x,16);
K2=0:length(X2)-1;
subplot(3,2,3);
stem(K2,abs(X2));
grid;
xlabel('length');
ylabel('x[n]');
subplot(3,2,4);
stem(K2,angle(X2));
grid;
xlabel('length');
ylabel('x[n]');
X3=fft(x,64);
K3=0:length(X3)-1;
subplot(3,2,5);
stem(K3,abs(X3));
grid;
xlabel('length');
ylabel('x[n]');
subplot(3,2,6);
stem(K3,angle(X3));
grid;
xlabel('length');
ylabel('x[n]');
 

experiment 4

1)clc;
clear; 
close all;
y=[1 2 3 -2 6 4];
N=8;
l=length(y);
s=N-l;
X=[y zeros(1,s)];
n=0:N-1;
z=X(mod(-n,N)+1);


2)clc;
clear; 
close all;
x=[1 3 -2 0 5 7 2 -1];
m=3;
N=8;
s=0:N-1;
X=x(mod(s-m,N)+1);
Y=x(mod(s+m,N)+1);
subplot(2,1,1),stem(s,X);
subplot(2,1,2),stem(s,Y);


3)clc;
clear;
close all;
x=[1 -4 2 6 -3 7 4 1];
N=8;
n=0:N-1;
y=x(mod(-n,N)+1);
a=((x+y)/2);
b=((x-y)/2);
subplot(2,1,1),stem(n,a);
xlabel('N');
ylabel('a');
legend('even part');
grid;
subplot(2,1,2),stem(n,b);
xlabel('N');
ylabel('b');
legend('odd part');
grid;


4)clc;
clear;
close all;
x=[1 2 3 4];
X=fft(x);
y=[4 1 2 3];
k=0:3;
N=4;
A=exp((-1j*2*pi*k)/N);
Y=(A).*X;
Z=fft(y);
if round(Y)==round(Z);
 disp('time shift property is verified');
else
 disp('time shift property is not verified')
end


5)clc;
clear;
close all;
x=[1 -2 3 -5 -7 6 4 5 9 12 -4];
N=16;
l=length(x);
d=N-1;
X=[x zeros(1,d)];
n=0:length(x)-1;
X=fft(x);
Y=x(mod(-n,length(x))+1);
Z=fft(Y);
A=X(mod(n,length(x))+1);
if round(Z)==round(A);
 disp('time reversal property is verified');
else
 disp('time reversal property is not verified');
end


6)clc;
close;
clear;
x=[5 4 3 2 2 4 6 6 8];
l=length(x);
h=[4 3 2 1 ];
m=length(h);
X=conv(x,h);
h1=[h zeros(1,12-m)];
x1=[x zeros(1,12-l)];
z=ifft(fft(x1).*fft(h1));
if round(z)==round(X);
 disp('linear convolution property is verfied');
else
 disp('linear convolution property is not 
verfied');
end;


7)clc;
clear;
close;
x = [1 5 3 7 -20 12 8 2];
xmod = abs(x);
xmodsquared = xmod.^2;
time_domain = sum(xmodsquared)
N = length(x);
f = fft(x,N);
fmod = abs(f);
fmodsquared = fmod.^2;
frequency_domain = sum(fmodsquared)/N
if round(time_domain)==round(frequency_domain);
 disp('parsevals property is verfied');
else
 disp('parsevals property is not verfied');
end;


8)clc;
clear all;
close all;
X=[0,-4,0,1,0,-1,0,4];
N=8;
Y=fft(X,N);
disp(Y);


experiment 5

1)clc;
clear all;
close all;
x=[5 4 3 2 2 4 6 8];
disp('x'); disp(x);
h=[4 3 2 1 0 0 0 0];
disp('h'); disp(h);
n=0:length(x)-1;
N=8;
y=(x)+(1i*h);
Y=fft(y);
yr=Y(mod(-n,N)+1);
z=conj(yr);
X=(Y+z)/2;
disp('X'); disp(X);
a=(Y-z)/2;
disp('a'); disp(a);


2)clc;
clear all;
close all;
n=0:1:15;
k=0:1:7;
g=[0 0 0 3 4 5 6 7 8 9 10 11 0 0 0 0];
disp('G'); disp(fft(g));
w=exp((-1i*2*pi)/16);
N=16;
G=fft(g);
X=[0 0 4 6 8 10 0 0];
Y=[0 3 5 7 9 11 0 0];
x=fft(X);
y=fft(Y);
A=x+(w.^k).*(y);
B=x-(w.^k).*(y);
C=[A B];
disp('G[k]'); disp(C);


experiment 6

1)clc;
clear;
close all;
x=[1,2,-1,2,3,-2,-3,-1,1,1,2,-1];
h=[1,2,3,-1];
L=6;
x1=[zeros(1,length(h)-1),x(1),x(2),x(3),x(4),x(5),x(6)];
l=length(x1);
h1=[h,zeros(1,l-length(h))];
x2=[x(4),x(5),x(6),x(7),x(8),x(9),x(10),x(11),x(12)];
x3=[x(10),x(11),x(12),zeros(1,l-(length(h)-1))];
l=length(x1);
h1=[h,zeros(1,l-length(h))];
y1=cconv(x1,h1);
y2=cconv(x2,h1);
y3=cconv(x3,h1);
y=[y1(4),y1(5),y1(6),y1(7),y1(8),y1(9),y2(4),y2(5),y2(6),y2(7),y2(8),y2(9),y3(4),y3(5),y3(6),y3(7),y3(8),y3(9)];
disp('The block convolution of x(n) and h(n) using overlap save');
disp(y);
disp('verify');
disp(conv(x,h));


2)clc;
clear;
close all;
x=[1,2,-1,2,3,-2,-3,-1,1,1,2,-1];
h=[1,2,3,-1];
x1=[x(1),x(2),x(3),x(4),x(5),x(6),zeros(1,length(h)-1)];
x2=[x(7),x(8),x(9),x(10),x(11),x(12),zeros(1,length(h)-1)];
l=length(x1);
h1=[h,zeros(1,l-length(h))];
y1=cconv(x1,h1);
y2=cconv(x2,h1);
y=[y1(1),y1(2),y1(3),y1(4),y1(5),y1(6),y1(7)+y2(1),y1(8)+y2(2),y1(9)+y2(3),y2(4),y2(5),y2(6),y2(7),y2(8),y2(9)];
disp('The block convolution of x(n) and h(n) using overlap add');
disp(y);
disp(conv(x,h));


experiment7

1)clc;
clear all;
close all;
fp=400;
fs=800;
fsamp=2000;
kp=0.4;
ks=30;
wp=fp/(fsamp/2);
ws=fs/(fsamp/2);
[N,wc]=buttord(wp,ws,kp,ks);
[b,a]=butter(N,wc,'low');
freqz(b,a);
title('low pass butterworth filter');
[b,a]=butter(N,wc,'high');
figure;
freqz(b,a);
title('high pass butterworh filter');
wn=[wp ws];
[b,a]=butter(N,wn,'stop');
figure;
freqz(b,a);
title('bandstop butterworth filter');
[b,a]=butter(N,wn,'bandpass');
figure;
freqz(b,a);
title('bandpass butterworth filter');


2) clc;
close;
clear;
Wp=[0.2 0.4];
Ws=[0.1 0.5];
kp=2;
ks=20;
[n,Wp] = cheb1ord(Wp,Ws,kp,ks); 
[b,a] = cheby1(n,kp,Wp,'bandpass'); 
freqz(b,a);
title('bandpass chebyshev');
[n,Wp] = cheb1ord(Wp,Ws,kp,ks); 
[b,a] = cheby1(n,kp,Wp,'stop');
figure;
freqz(b,a);
title('bandstop chebyshev');


experiment 8

1)clc;
clear;
close;
N = 21;
Wc = 0.6*pi;
n = 0:N-1;
alpha = (N-1)/2;
z = n-alpha+eps;
hd = (sin(Wc*z))./(pi*z);
Wr = boxcar(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(1);
subplot(2,1,1);
freqz(h, 1, W);title('rectangular');
figure(2);
freqz(Wr, 1, W);title('rectangular');
hd = (sin(Wc*z))./(pi*z);
Wr = hamming(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(3);
subplot(2,1,1);
freqz(h, 1, W);title('hamming');
figure(4);
freqz(Wr, 1, W);
title('hamming');
hd = (sin(Wc*z))./(pi*z);
Wr = hanning(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(5);
subplot(2,1,1);
freqz(h, 1, W);
title('hamming');
figure(6);
freqz(Wr, 1, W);
title('hanning');
hd = (sin(Wc*z))./(pi*z);
Wr = blackman(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(7);
subplot(2,1,1);
freqz(h, 1, W);title('blackman');
figure(8);
freqz(Wr, 1, W);
title('blackman');
hd = (sin(Wc*z))./(pi*z);
Wr = bartlett(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(9);
subplot(2,1,1);
freqz(h, 1, W);
title('bartlett');
figure(10);
freqz(Wr, 1, W);
title('bartlett');


2)clc;
clear;
close;
N = 21;
Wc = 0.3*pi;
n = 0:N-1;
alpha = (N-1)/2;
z = n-alpha+eps;
hd = (sin(pi*z)-(sin(Wc*z)))./(pi*z);
Wr = boxcar(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(1);
subplot(2,1,1);
freqz(h, 1, W);
title('rectangular');
figure(2);
freqz(Wr, 1, W);
title('rectangular');
hd = (sin(pi*z)-(sin(Wc*z)))./(pi*z);
Wr = hamming(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(3);
subplot(2,1,1);
freqz(h, 1, W);
title('hamming');
figure(4);
freqz(Wr, 1, W);
title('hamming');
hd = (sin(pi*z)-(sin(Wc*z)))./(pi*z);
Wr = hanning(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(5);
subplot(2,1,1);
freqz(h, 1, W);
title('hanning');
figure(6);
freqz(Wr, 1, W);
title('hanning');
hd = (sin(pi*z)-(sin(Wc*z)))./(pi*z);
Wr = blackman(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(7);
subplot(2,1,1);
freqz(h, 1, W);
title('blackman');
figure(8);
freqz(Wr, 1, W);
title('blackman');
hd = (sin(pi*z)-(sin(Wc*z)))./(pi*z);
Wr = bartlett(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(9);
subplot(2,1,1);
freqz(h, 1, W);
title('bartlett');
figure(10);
freqz(Wr, 1, W);
title('bartlett');



3)clc;
clear;
close;
N = 21;
Wc1 = 0.3*pi;
Wc2 = 0.8*pi;
n = 0:N-1;
alpha = (N-1)/2;
z = n-alpha+eps;
hd = (sin(Wc2*z) - sin(Wc1*z))./(pi*z);
Wr = boxcar(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(1);
subplot(2,1,1);
freqz(h, 1, W);title('rectangular');
figure(2);
freqz(Wr, 1, W);title('rectangular');
hd = (sin(Wc2*z) - sin(Wc1*z))./(pi*z);
Wr = hamming(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(3);
subplot(2,1,1);
freqz(h, 1, W);title('hamming');
figure(4);
freqz(Wr, 1, W);title('hamming');
hd = (sin(Wc2*z) - sin(Wc1*z))./(pi*z);
Wr = hanning(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(5);
subplot(2,1,1);
freqz(h, 1, W);title('hanning');
figure(6);
freqz(Wr, 1, W);title('hanning');
hd = (sin(Wc2*z) - sin(Wc1*z))./(pi*z);
Wr = blackman(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(7);
subplot(2,1,1);
freqz(h, 1, W);title('blackman');
figure(8);
freqz(Wr, 1, W);title('blackman');
hd = (sin(Wc2*z) - sin(Wc1*z))./(pi*z);
Wr = bartlett(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(9);
subplot(2,1,1);
freqz(h, 1, W);title('bartlett');
figure(10);
freqz(Wr, 1, W);title('bartlett');


4)clc;
clear;
close;
N = 21;
Wc1 = 0.4*pi;
Wc2 = 0.7*pi;
n = 0:N-1;
alpha = (N-1)/2;
z = n-alpha+eps;
hd = (sin(Wc1*z) - sin(Wc2*z)+sin(pi*z))./(pi*z);
Wr = boxcar(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(1);
subplot(2,1,1);
freqz(h, 1, W);
title('rectangular');
figure(2);
freqz(Wr, 1, W);
title('rectangular');
hd = (sin(Wc1*z) - sin(Wc2*z)+sin(pi*z))./(pi*z);
Wr = hamming(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(3);
subplot(2,1,1);
freqz(h, 1, W);
title('hamming');
figure(4);
freqz(Wr, 1, W);
title('hamming');
hd = (sin(Wc1*z) - sin(Wc2*z)+sin(pi*z))./(pi*z);
Wr = hanning(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(5);
subplot(2,1,1);
freqz(h, 1, W);
title('hanning');
figure(6);
freqz(Wr, 1, W);
title('hanning');
hd = (sin(Wc1*z) - sin(Wc2*z)+sin(pi*z))./(pi*z);
Wr = blackman(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(7);
subplot(2,1,1);
freqz(h, 1, W);
title('blackman');
figure(8);
freqz(Wr, 1, W);
title('blackman');
hd = (sin(Wc1*z) - sin(Wc2*z)+sin(pi*z))./(pi*z);
Wr = bartlett(N);
h = hd.*Wr';
W = 0:0.01:pi;
figure(9);
subplot(2,1,1);
freqz(h, 1, W);
title('bartlett');
figure(10);
freqz(Wr, 1, W);
title('bartlett');


experiment 9

1)% Chebyshev filter 
No=4; 
t=0:0.00001:1/800; 
xt=2*sin(100*pi*t)+3*cos(800*pi*t)+sin(1600*pi*t); 
fs=2000; 
N1=2*pi/(100*pi/fs); 
N2=2*pi/(800*pi/fs); 
N3=4*pi/(1600*pi/fs); 
N=lcm(lcm(N1,N2),N3); 
n=0:N-1; 
xn=2*sin(100*pi*n/fs)+3*cos(800*pi*n/fs)+sin(1600*pi*n/fs); 
subplot(2,2,1) 
stem(n,xn);title('x(n)') 
xlabel('n');ylabel('Amplitude');grid 
k=0:N-1; 
Xn=fft(xn); 
subplot(2,2,3) 
stem(k*fs/N,abs(Xn));title('X(k)') 
xlabel('fa (Hz)');ylabel('Magnitude');grid 
w1=2*250/fs; 
w2=2*600/fs; 
wb=[w1 w2]; 
[b,a]=cheby1(No,0.2,wb); 
y=filter(b,a,xn); 
subplot(2,2,2) 
stem(n,y);title('y(n)') 
xlabel('n');ylabel('Amplitude');grid 
Y=fft(y); 
subplot(2,2,4) 
stem(k*fs/N,abs(Y));title('Y(k)') 
xlabel('fa (Hz)');ylabel('Magnitude');grid 
figure 
freqz(b,a);
title('Chebyshev Bandpass’)


2)% Butterworth filter bandstop 
No=6;
t=0:0.00001:1/200; 
xt=2+sin(80*pi*t)+sin(200*pi*t); 
fs=800; 
N1=2*pi/(80*pi/fs); 
N2=round(2*pi/(200*pi/fs)); 
N=lcm(N1,N2); 
n=0:N-1; 
xn=2+sin(80*pi*n/fs)+sin(200*pi*n/fs); 
figure 
subplot(2,2,1) 
stem(n,xn);title('x(n)') 
xlabel('n');ylabel('Amplitude');grid 
k=0:N-1; 
Xn=fft(xn); 
subplot(2,2,3) 
stem(k*fs/N,abs(Xn));title('X(k)') 
xlabel('fa (Hz)');ylabel('Magnitude');grid 
w1=2*20/fs; 
w2=2*70/fs; 
wb=[w1 w2]; 
[b,a]=butter(No,wb,'stop'); 
y=filter(b,a,xn); 
subplot(2,2,2) 
stem(n,y);title('y(n)') 
xlabel('n');ylabel('Amplitude');grid 
Y=fft(y); 
subplot(2,2,4) 
stem(k*fs/N,abs(Y));title('Y(k)') 
xlabel('fa (Hz)');ylabel('Magnitude');grid 
figure 
freqz(b,a);title('Butterworth Band Stop')
