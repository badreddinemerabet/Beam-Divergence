%% beam_expansion_animation.m
% Animate transverse beam profiles and characteristic expansion over 0-100 km.
clear; close all; clc;
lambda=1064e-9; k=2*pi/lambda; P=1; w0=10e-3; zr=pi*w0^2/lambda;
besselTheta=0.05e-3; kr=k*sin(besselTheta); besselCore=2.405/kr; besselZmax=0.20/tan(besselTheta);
airyX0=0.05; airyA=0.08; airyZ0=k*airyX0^2; topHatR=w0;
zvals=linspace(0,100e3,41); x=linspace(-12,12,1000);
names={'Bessel-Gaussian diagnostic','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10'};
outdir=fullfile(fileparts(mfilename('fullpath')),'outputs_other_beams'); if ~exist(outdir,'dir'); mkdir(outdir); end
video=VideoWriter(fullfile(outdir,'beam_expansion_animation_matlab.mp4'),'MPEG-4'); video.FrameRate=6; open(video);
figure('Color','w','Position',[100 100 1400 950]);
for i=1:numel(zvals)
    z=zvals(i); w=w0*sqrt(1+(z/zr)^2); rho=abs(x);
    profiles=normalized_beam_fields(x,zeros(size(x)),z,lambda,P,w0);
    clf; tiledlayout(3,2,'Padding','compact');
    for qn=1:5; nexttile; semilogy(x,profiles{qn},'LineWidth',1.8); xlim([-12 12]); ylim([1e-10 2e4]); grid on; title(names{qn}); xlabel('transverse coordinate (m)'); ylabel('I (W/m^2)'); end
    nexttile; hold on; widths=[max(besselCore,0.20*(z>besselZmax)),max(airyX0,abs(z/airyZ0)^2*airyX0),max(topHatR,1.22*lambda*max(z,eps)/(2*topHatR)),w,w];
    plot(zvals(1:i)/1000,repmat(widths(1),1,i),'b','LineWidth',1.5); plot(zvals(1:i)/1000,repmat(widths(2),1,i),'Color',[0.9 .3 .1],'LineWidth',1.5); plot(zvals(1:i)/1000,repmat(widths(3),1,i),'g','LineWidth',1.5); plot(zvals(1:i)/1000,w0*sqrt(1+(zvals(1:i)/zr).^2),'m','LineWidth',1.5); plot(zvals(1:i)/1000,w0*sqrt(1+(zvals(1:i)/zr).^2),'c','LineWidth',1.5); xlim([0 100]); ylim([0 40]); grid on; xlabel('propagation distance (km)'); ylabel('characteristic width (m)'); legend(names,'FontSize',7,'Location','northwest'); title('Expansion envelope');
    sgtitle(sprintf('Beam expansion during propagation: z = %.1f km',z/1000)); drawnow; writeVideo(video,getframe(gcf));
end
close(video); fprintf('Wrote %s\n',fullfile(outdir,'beam_expansion_animation_matlab.mp4'));

