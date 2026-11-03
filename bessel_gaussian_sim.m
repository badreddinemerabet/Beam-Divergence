%% bessel_gaussian_sim.m
% Finite-power Bessel-Gaussian beam over 0-100 km.
clear; close all; clc;
lambda=1064e-9; k=2*pi/lambda; P=1; coneAngle=0.05e-3; kr=k*sin(coneAngle);
envelopeW0=0.20; zMax=100e3; zrEnv=pi*envelopeW0^2/lambda; coreRadius=2.405/kr;
z=linspace(0,zMax,1001); wEnv=envelopeW0*sqrt(1+(z/zrEnv).^2);

autoNorm=@(w) P/(2*pi*trapz(linspace(0,8*w,25000), (besselj(0,kr*linspace(0,8*w,25000)).^2).*exp(-2*linspace(0,8*w,25000).^2/w.^2).*linspace(0,8*w,25000)));
Iaxis=zeros(size(z)); for n=1:numel(z); Iaxis(n)=autoNorm(wEnv(n)); end
fprintf('Bessel-Gaussian: lambda=%.0f nm, P=%.1f W\n',lambda*1e9,P); fprintf('Core radius=%.2f mm; envelope radius: %.3f m -> %.3f m\n',coreRadius*1e3,envelopeW0,wEnv(end));

outdir=fullfile(fileparts(mfilename('fullpath')),'outputs_bessel_gaussian'); if ~exist(outdir,'dir'); mkdir(outdir); end
T=table(z(:),wEnv(:),Iaxis(:),'VariableNames',{'z_m','envelope_radius_m','on_axis_irradiance_W_m2'}); writetable(T,fullfile(outdir,'bessel_gaussian_propagation_matlab.csv'));
params=table({'wavelength';'power';'cone_angle';'core_radius';'envelope_radius_z0';'envelope_radius_100km';'envelope_rayleigh_range'},[lambda*1e9;P;coneAngle*1e3;coreRadius*1e3;envelopeW0;wEnv(end);zrEnv/1000],{'nm';'W';'mrad';'mm';'m';'m';'km'},'VariableNames',{'parameter','value','units'}); writetable(params,fullfile(outdir,'bessel_gaussian_parameters_matlab.csv')); save(fullfile(outdir,'bessel_gaussian_results.mat'),'T','params');

% Propagation map
zmap=linspace(0,zMax,600); y=linspace(-0.6,0.6,500); [Z,Y]=meshgrid(zmap,y); W=envelopeW0*sqrt(1+(Z/zrEnv).^2); Am=zeros(size(zmap)); for n=1:numel(zmap); Am(n)=autoNorm(envelopeW0*sqrt(1+(zmap(n)/zrEnv)^2)); end
F=zeros(size(Z)); for n=1:numel(zmap); F(:,n)=Am(n)*besselj(0,kr*abs(y)).^2.*exp(-2*y.^2/W(1,n)^2); end
figure('Color','w'); imagesc(zmap/1000,y,log10(max(F,1e-10))); axis xy; colorbar; xlabel('z (km)'); ylabel('transverse y (m)'); title('Bessel-Gaussian propagation cross-section, log10 irradiance'); saveas(gcf,fullfile(outdir,'bessel_gaussian_propagation_map_matlab.png'));

% Transverse maps at z=0 and z=100 km
for pass=1:2
    if pass==1; zz=0; tag='z0'; extent=0.25; else; zz=zMax; tag='z100km'; extent=0.6; end
    axisv=linspace(-extent,extent,600); [X,Y]=meshgrid(axisv,axisv); ww=envelopeW0*sqrt(1+(zz/zrEnv)^2); AA=autoNorm(ww); R=sqrt(X.^2+Y.^2); FF=AA*besselj(0,kr*R).^2.*exp(-2*R.^2/ww^2);
    figure('Color','w'); imagesc(axisv,axisv,log10(max(FF,1e-10))); axis xy equal tight; colorbar; xlabel('x (m)'); ylabel('y (m)'); title(sprintf('Bessel-Gaussian transverse irradiance at z=%g km',zz/1000)); saveas(gcf,fullfile(outdir,['bessel_gaussian_transverse_' tag '_matlab.png']));
end
