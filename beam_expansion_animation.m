%% beam_expansion_animation.m
% Animate transverse beam profiles and characteristic expansion over 0-100 km.
clear; close all; clc;
lambda=1064e-9; k=2*pi/lambda; P=1; w0=10e-3; zr=pi*w0^2/lambda;
besselTheta=0.05e-3; kr=k*sin(besselTheta); besselCore=2.405/kr; besselZmax=0.20/tan(besselTheta);
airyX0=0.05; airyA=0.08; airyZ0=k*airyX0^2; topHatR=w0;
zvals=linspace(0,100e3,41); x=linspace(-12,12,1000);
names={'Bessel J0','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10'};
outdir=fullfile(fileparts(mfilename('fullpath')),'outputs_other_beams'); if ~exist(outdir,'dir'); mkdir(outdir); end
video=VideoWriter(fullfile(outdir,'beam_expansion_animation_matlab.mp4'),'MPEG-4'); video.FrameRate=6; open(video);
figure('Color','w','Position',[100 100 1400 950]);
for i=1:numel(zvals)
    z=zvals(i); w=w0*sqrt(1+(z/zr)^2); rho=abs(x);
    b=2*P/(pi*besselCore^2)*besselj(0,kr*rho).^2; if z>besselZmax; b=b*exp(-((z-besselZmax)/besselZmax)^2); end
    s=x/airyX0; xi=z/airyZ0; af=airy(s-xi^2/4+1i*airyA*xi); a=P/(sqrt(2*pi)*airyX0)*abs(af).^2.*exp(2*airyA*s-airyA*xi^2);
    ztr=k*topHatR^2/2; q=k*topHatR*rho/max(z,eps); q(q==0)=eps; disk=(2*besselj(1,q)./q).^2; near=P/(pi*topHatR^2)*(rho<=topHatR); far=P/(pi*topHatR^2)*(ztr/max(z,eps))^2.*disk; blend=min(max(z/ztr,0),1); t=(1-blend)*near+blend*far;
    rho2=2*x.^2/w^2; lg=4*P/(pi*w^2)*rho2.*exp(-rho2); u=sqrt(2)*x/w; hg=2*(2*P/(pi*w^2))*u.^2.*exp(-u.^2); profiles={b,a,t,lg,hg};
    clf; tiledlayout(3,2,'Padding','compact');
    for qn=1:5; nexttile; semilogy(x,profiles{qn},'LineWidth',1.8); xlim([-12 12]); ylim([1e-10 2e4]); grid on; title(names{qn}); xlabel('transverse coordinate (m)'); ylabel('I (W/m^2)'); end
    nexttile; hold on; widths=[max(besselCore,0.20*(z>besselZmax)),max(airyX0,abs(z/airyZ0)^2*airyX0),max(topHatR,1.22*lambda*max(z,eps)/(2*topHatR)),w,w];
    plot(zvals(1:i)/1000,repmat(widths(1),1,i),'b','LineWidth',1.5); plot(zvals(1:i)/1000,repmat(widths(2),1,i),'Color',[0.9 .3 .1],'LineWidth',1.5); plot(zvals(1:i)/1000,repmat(widths(3),1,i),'g','LineWidth',1.5); plot(zvals(1:i)/1000,w0*sqrt(1+(zvals(1:i)/zr).^2),'m','LineWidth',1.5); plot(zvals(1:i)/1000,w0*sqrt(1+(zvals(1:i)/zr).^2),'c','LineWidth',1.5); xlim([0 100]); ylim([0 40]); grid on; xlabel('propagation distance (km)'); ylabel('characteristic width (m)'); legend(names,'FontSize',7,'Location','northwest'); title('Expansion envelope');
    sgtitle(sprintf('Beam expansion during propagation: z = %.1f km',z/1000)); drawnow; writeVideo(video,getframe(gcf));
end
close(video); fprintf('Wrote %s\n',fullfile(outdir,'beam_expansion_animation_matlab.mp4'));
