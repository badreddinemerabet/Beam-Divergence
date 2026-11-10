%% gaussian_beam_sim.m
% Free-space Gaussian-beam divergence simulation from 0 to 100 km.
% Baseline: 1064 nm, 1550 nm, 3.5 um, 10.6 um; w0 = 10 mm; P = 1 W; M2 = 1.
% Edit the parameter block below for other studies.
clear; close all; clc;
set(groot,'defaultFigureColormap',turbo(256));

%% Parameters
names = {'Near-IR 1064 nm','Near-IR 1550 nm','Mid-IR 3.5 um','Mid-IR 10.6 um'};
lambda_nm = [1064, 1550, 3500, 10600];
w0_mm = [10, 10, 10, 10];             % 1/e^2 intensity radius at z=0
P_W = [1, 1, 1, 1];                    % total optical power
M2 = [1, 1, 1, 1];                     % beam quality factor
colors = [217 119 6; 234 179 8; 8 145 178; 220 38 38] / 255;

% Propagation and plotting grids
z = linspace(0, 100e3, 1001);           % 0 to 100 km, metres
r = linspace(0, 0.5, 600);              % radial coordinate at waist, metres

% Physical constants
c0 = 299792458;
eps0 = 8.8541878128e-12;
mu0 = 4*pi*1e-7;
eta0 = sqrt(mu0/eps0);

%% Beam parameters
lambda_m = lambda_nm * 1e-9;
w0_m = w0_mm * 1e-3;
zR_m = pi .* w0_m.^2 ./ (M2 .* lambda_m);
theta_rad = M2 .* lambda_m ./ (pi .* w0_m); % far-field half-angle

fprintf('\nFree-space Gaussian-beam summary\n');
fprintf('w0 = 10 mm; P = 1 W; M^2 = 1\n\n');
fprintf('%-20s %10s %12s %19s %19s %16s\n', ...
    'Beam','lambda','zR (m)','half-angle (mrad)','full-angle (mrad)','w(100 km) (m)');
fprintf('%s\n', repmat('-',1,95));
for k = 1:numel(names)
    w100 = w0_m(k) * sqrt(1 + (100e3/zR_m(k))^2);
    fprintf('%-20s %10.1f %12.3f %19.5f %19.5f %16.3f\n', names{k}, ...
        lambda_nm(k), zR_m(k), 1e3*theta_rad(k), 2e3*theta_rad(k), w100);
end

%% Compute and export propagation table
nZ = numel(z);
nB = numel(names);
beamCol = strings(nZ*nB,1); lambdaCol = zeros(nZ*nB,1); zCol = zeros(nZ*nB,1);
wCol = zeros(nZ*nB,1); ICol = zeros(nZ*nB,1); ECol = zeros(nZ*nB,1); HCol = zeros(nZ*nB,1); PCol = zeros(nZ*nB,1);
row = 1;
for k = 1:nB
    w_z = w0_m(k) .* sqrt(1 + (z./zR_m(k)).^2);
    I_axis = 2*P_W(k) ./ (pi*w_z.^2);
    E_axis = sqrt(2*I_axis/(c0*eps0));
    H_axis = E_axis/eta0;
    idx = row:(row+nZ-1);
    beamCol(idx) = names{k}; lambdaCol(idx) = lambda_nm(k); zCol(idx) = z(:);
    wCol(idx) = w_z(:); ICol(idx) = I_axis(:); ECol(idx) = E_axis(:); HCol(idx) = H_axis(:); PCol(idx) = P_W(k);
    row = row+nZ;
end
propagation = table(beamCol, lambdaCol, zCol, wCol, ICol, ECol, HCol, PCol, ...
    'VariableNames', {'beam','wavelength_nm','z_m','radius_m','on_axis_irradiance_W_m2','on_axis_E_V_m','on_axis_H_A_m','total_power_W'});
summary = table(string(names(:)), lambda_nm(:), w0_mm(:), P_W(:), M2(:), zR_m(:), theta_rad(:), ...
    1e3*theta_rad(:), 2e3*theta_rad(:), w0_m(:).*sqrt(1+(100e3./zR_m(:)).^2), ...
    'VariableNames', {'beam','wavelength_nm','waist_radius_mm','power_W','M2','rayleigh_range_m', ...
    'half_angle_rad','half_angle_mrad','full_angle_mrad','radius_at_100km_m'});

outdir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~exist(outdir, 'dir'); mkdir(outdir); end
writetable(propagation, fullfile(outdir, 'beam_propagation_matlab.csv'));
writetable(summary, fullfile(outdir, 'beam_summary_matlab.csv'));
save(fullfile(outdir, 'beam_results.mat'), 'propagation', 'summary', 'names', 'lambda_nm', 'w0_mm', 'P_W', 'M2');

%% Main comparison plots
figure('Color','w','Position',[100 100 1250 800]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
for k = 1:nB
    w_z = w0_m(k) .* sqrt(1 + (z./zR_m(k)).^2);
    I_axis = 2*P_W(k) ./ (pi*w_z.^2);
    I_r = 2*P_W(k)/(pi*w0_m(k)^2) .* exp(-2*r.^2/w0_m(k)^2);
    E_axis = sqrt(2*I_axis/(c0*eps0));
    nexttile(1); plot(z/1000,w_z,'LineWidth',1.8,'Color',colors(k,:)); hold on;
    nexttile(2); semilogy(z/1000,I_axis,'LineWidth',1.8,'Color',colors(k,:)); hold on;
    nexttile(3); plot(r*1000,I_r,'LineWidth',1.8,'Color',colors(k,:)); hold on;
    nexttile(4); plot(z/1000,E_axis,'LineWidth',1.8,'Color',colors(k,:)); hold on;
end
labels = {'1/e^2 beam radius','On-axis irradiance','Transverse irradiance at waist','On-axis electric-field amplitude'};
xlabels = {'Propagation distance z (km)','Propagation distance z (km)','Radius r (mm)','Propagation distance z (km)'};
ylabels = {'w(z) (m)','I(0,z) (W/m^2)','I(r,0) (W/m^2)','|E(0,z)| (V/m)'};
for q = 1:4
    nexttile(q); title(labels{q}); xlabel(xlabels{q}); ylabel(ylabels{q}); grid on; legend(names,'Location','best','FontSize',8);
end
sgtitle('Free-space wavelength-dependent Gaussian beams: 0–100 km');
saveas(gcf, fullfile(outdir, 'beam_comparison_matlab.png'));

%% 2-D irradiance cross-sections
zmap = linspace(0,100e3,500);
y = linspace(-120,120,500); % metres
[Z,Y] = meshgrid(zmap,y);
figure('Color','w','Position',[100 100 1250 800]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
for k = 1:nB
    wmap = w0_m(k).*sqrt(1+(Z./zR_m(k)).^2);
    Imap = 2*P_W(k)./(pi*wmap.^2).*exp(-2*Y.^2./wmap.^2);
    nexttile; imagesc(zmap/1000,y,Imap); axis xy; colorbar; colormap turbo;
    title(names{k}); xlabel('z (km)'); ylabel('transverse y (m)');
end
sgtitle('2-D irradiance cross-sections (absolute scale)');
saveas(gcf, fullfile(outdir, 'beam_intensity_maps_matlab.png'));

fprintf('\nMATLAB plots and tables saved to: %s\n', outdir);

