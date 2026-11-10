%% transverse_irradiance_maps.m
% True transverse x-y irradiance maps for the five non-Gaussian beams.
clear; close all; clc;
set(groot,'defaultFigureColormap',turbo(256));
lambda=1064e-9; k=2*pi/lambda; P=1; w0=10e-3; zr=pi*w0^2/lambda;
besselTheta=0.05e-3; kr=k*sin(besselTheta); besselCore=2.405/kr; besselZmax=0.20/tan(besselTheta);
airyX0=0.05; airyA=0.08; airyZ0=k*airyX0^2; topHatR=10e-3;
names={'Bessel-Gaussian diagnostic','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10'};
outdir=fullfile(fileparts(mfilename('fullpath')),'outputs_other_beams'); if ~exist(outdir,'dir'); mkdir(outdir); end

for pass=1:2
    if pass==1; z=0; tag='z0'; extent=0.15; else; z=100e3; tag='z100km'; extent=10; end
    axisv=linspace(-extent,extent,500); [X,Y]=meshgrid(axisv,axisv); rho=sqrt(X.^2+Y.^2); w=w0*sqrt(1+(z/zr)^2);
    fields=normalized_beam_fields(X,Y,z,lambda,P,w0); figure('Color','w','Position',[100 100 1350 950]); tiledlayout(2,3,'Padding','compact');
    for qn=1:5; nexttile; imagesc(axisv,axisv,log10(max(fields{qn},1e-10))); axis xy equal tight; colorbar; title(names{qn}); xlabel('x (m)'); ylabel('y (m)'); end
    nexttile; axis off; sgtitle(sprintf('Transverse 2-D irradiance maps at z = %s',tag)); saveas(gcf,fullfile(outdir,['transverse_irradiance_maps_' tag '_matlab.png']));
end

