%% transverse_irradiance_maps.m
% True transverse x-y irradiance maps for the five non-Gaussian beams.
clear; close all; clc;
lambda=1064e-9; k=2*pi/lambda; P=1; w0=10e-3; zr=pi*w0^2/lambda;
besselTheta=0.05e-3; kr=k*sin(besselTheta); besselCore=2.405/kr; besselZmax=0.20/tan(besselTheta);
airyX0=0.05; airyA=0.08; airyZ0=k*airyX0^2; topHatR=10e-3;
names={'Bessel J0','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10'};
outdir=fullfile(fileparts(mfilename('fullpath')),'outputs_other_beams'); if ~exist(outdir,'dir'); mkdir(outdir); end

for pass=1:2
    if pass==1; z=0; tag='z0'; extent=0.15; else; z=100e3; tag='z100km'; extent=10; end
    axisv=linspace(-extent,extent,500); [X,Y]=meshgrid(axisv,axisv); rho=sqrt(X.^2+Y.^2); w=w0*sqrt(1+(z/zr)^2);
    bessel=2*P/(pi*besselCore^2)*besselj(0,kr*rho).^2; if z>besselZmax; bessel=bessel*exp(-((z-besselZmax)/besselZmax)^2); end
    sX=X/airyX0; sY=Y/airyX0; xi=z/airyZ0;
    ax=airy(sX-xi^2/4+1i*airyA*xi); ay=airy(sY-xi^2/4+1i*airyA*xi);
    airy2d=P/(sqrt(2*pi)*airyX0)*abs(ax.*ay).^2.*exp(airyA*(sX+sY)-airyA*xi^2);
    ztr=k*topHatR^2/2; q=k*topHatR*rho/max(z,eps); q(q==0)=eps;
    airyDisk=(2*besselj(1,q)./q).^2; near=P/(pi*topHatR^2)*(rho<=topHatR); far=P/(pi*topHatR^2)*(ztr/max(z,eps))^2.*airyDisk; blend=min(max(z/ztr,0),1); topHat=(1-blend)*near+blend*far;
    rho2=2*rho.^2/w^2; lg01=4*P/(pi*w^2)*rho2.*exp(-rho2); u=sqrt(2)*X/w; hg10=2*(2*P/(pi*w^2))*u.^2.*exp(-u.^2);
    fields={bessel,airy2d,topHat,lg01,hg10}; figure('Color','w','Position',[100 100 1350 950]); tiledlayout(2,3,'Padding','compact');
    for qn=1:5; nexttile; imagesc(axisv,axisv,log10(max(fields{qn},1e-10))); axis xy equal tight; colorbar; title(names{qn}); xlabel('x (m)'); ylabel('y (m)'); end
    nexttile; axis off; sgtitle(sprintf('Transverse 2-D irradiance maps at z = %s',tag)); saveas(gcf,fullfile(outdir,['transverse_irradiance_maps_' tag '_matlab.png']));
end
