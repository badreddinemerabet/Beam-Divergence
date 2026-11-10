%% Shared finite-power non-Gaussian diagnostics (not full diffraction).
clear; close all; clc;
set(groot,'defaultFigureColormap',turbo(256));
lambda=1064e-9; P=1; w0=.01;
names={'Bessel-Gaussian diagnostic','Airy 2-D','Top-hat diagnostic','LG01','HG10'};
outdir=fullfile(fileparts(mfilename('fullpath')),'outputs_other_beams'); if ~exist(outdir,'dir'); mkdir(outdir); end
z=linspace(0,100e3,1001); vals=normalized_beam_fields(zeros(size(z)),zeros(size(z)),z,lambda,P,w0);
T=table(z(:),vals{1}(:),vals{2}(:),vals{3}(:),vals{4}(:),vals{5}(:),'VariableNames',{'z_m','Bessel_Gaussian','Airy','Top_hat','LG01','HG10'});
writetable(T,fullfile(outdir,'other_beams_on_axis_matlab.csv'));
figure('Color','w'); hold on;
for j=1:5; semilogy(z/1000,max(vals{j},1e-12)); end
set(gca,'YScale','log'); xlabel('z (km)'); ylabel('I (W/m^2)'); legend(names); title('Finite-power analytical diagnostics');
saveas(gcf,fullfile(outdir,'other_beams_comparison_matlab.png'));
x=linspace(-12,12,700); [Z,X]=meshgrid(linspace(0,100e3,500),x);
fields=normalized_beam_fields(X,zeros(size(X)),Z,lambda,P,w0);
figure('Color','w'); tiledlayout(3,2);
for j=1:5; nexttile; imagesc(Z(1,:)/1000,x,log10(max(fields{j},1e-12))); axis xy; title(names{j}); xlabel('z (km)'); ylabel('x (m)'); colorbar; end
saveas(gcf,fullfile(outdir,'other_beams_propagation_maps_matlab.png'));
save(fullfile(outdir,'other_beams_results.mat'),'T');
