%% other_beams_sim.m
% Unified theoretical comparison of Bessel, Airy, top-hat, LG01, and HG10 beams.
% Scalar paraxial models, lambda=1064 nm, P=1 W, range=0-100 km.
clear; close all; clc;

lambda = 1064e-9; c = 299792458; k = 2*pi/lambda; P = 1; w0 = 10e-3;
z = linspace(0,100e3,1001); r = linspace(-0.15,0.15,900);
names = {'Bessel J0','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10'};
colors = [37 99 235;234 88 12;22 163 74;147 51 234;8 145 178]/255;

% Model parameters
besselTheta = 0.05e-3; kr = k*sin(besselTheta); besselAperture = 0.20;
besselZmax = besselAperture/tan(besselTheta); besselCore = 2.405/kr;
airyX0 = 0.05; airyA = 0.08; airyZ0 = k*airyX0^2;
tophatR = w0; zr = pi*w0^2/lambda;

% MATLAB tables and summary
fprintf('Other-beam simulation: lambda=1064 nm, P=1 W, range=0-100 km\n');
fprintf('Bessel finite length: %.2f km; Bessel core radius: %.2f mm\n',besselZmax/1000,besselCore*1000);
fprintf('Airy characteristic length: %.3f km\n',airyZ0/1000);

%% Analytic transverse profiles
wz = w0*sqrt(1+(z/zr).^2);
I_bessel_axis = (2*P/(pi*besselCore^2))*ones(size(z));
I_bessel_axis(z>besselZmax) = I_bessel_axis(z>besselZmax).*exp(-((z(z>besselZmax)-besselZmax)/besselZmax).^2);
xi = z/airyZ0;
airyAxis = airy(-xi.^2/4 + 1i*airyA*xi);
airyAxis = abs(airyAxis).^2*P/(sqrt(2*pi)*airyX0).*exp(-airyA*xi.^2);
ztr = k*tophatR^2/2;
I_top_axis = P/(pi*tophatR^2) * ones(size(z));
farMask = z>ztr; I_top_axis(farMask) = P/(pi*tophatR^2)*(ztr./z(farMask)).^2;
I_lg_axis = zeros(size(z)); I_hg_axis = zeros(size(z));

% Profiles at z=0 and z=100 km
profile0 = zeros(numel(r),5); profile100 = zeros(numel(r),5);
for q = 1:2
    if q==1; zz=0; else; zz=100e3; end
    ww = w0*sqrt(1+(zz/zr)^2);
    profileB = 2*P/(pi*besselCore^2)*besselj(0,kr*r).^2;
    s = r/airyX0; xx = zz/airyZ0;
    % finite-energy Airy approximation evaluated along x
    profileA = P/(sqrt(2*pi)*airyX0)*abs(airy(s-xx^2/4+1i*airyA*xx)).^2.*exp(2*airyA*s-airyA*xx^2);
    profileT = P/(pi*tophatR^2)*(abs(r)<=tophatR);
    if zz>0
        arg = k*tophatR*abs(r)/zz; arg(arg==0)=eps;
        profileT = P/(pi*tophatR^2)*(ztr/zz)^2.*(2*besselj(1,arg)./arg).^2;
    end
    rho = sqrt(2)*r/ww; profileLG = 4*P/(pi*ww^2)*rho.^2.*exp(-rho.^2);
    u = sqrt(2)*r/ww; profileHG = 2*(2*P/(pi*ww^2))*u.^2.*exp(-u.^2);
    if q==1; profile0=[profileB(:),profileA(:),profileT(:),profileLG(:),profileHG(:)];
    else; profile100=[profileB(:),profileA(:),profileT(:),profileLG(:),profileHG(:)]; end
end

%% Export data
outdir = fullfile(fileparts(mfilename('fullpath')),'outputs_other_beams'); if ~exist(outdir,'dir'); mkdir(outdir); end
T = table(z(:),I_bessel_axis(:),airyAxis(:),I_top_axis(:),I_lg_axis(:),I_hg_axis(:), ...
    'VariableNames',{'z_m','Bessel_J0','Airy','Top_hat_aperture','LG01','HG10'});
writetable(T,fullfile(outdir,'other_beams_on_axis_matlab.csv'));
params = table({'wavelength';'power';'Gaussian_waist_radius';'Bessel_cone_angle';'Bessel_finite_length';'Bessel_core_radius';'Airy_scale';'top_hat_radius'}, ...
    [lambda*1e9;P;w0*1e3;besselTheta*1e3;besselZmax/1000;besselCore*1e3;airyX0*1e3;tophatR*1e3], ...
    {'nm';'W';'mm';'mrad';'km';'mm';'mm';'mm'},'VariableNames',{'parameter','value','units'});
writetable(params,fullfile(outdir,'other_beams_parameters_matlab.csv')); save(fullfile(outdir,'other_beams_results.mat'),'T','params');

%% Comparison plots
figure('Color','w','Position',[100 100 1300 850]); tiledlayout(2,2,'Padding','compact');
nexttile; semilogy(z/1000,I_bessel_axis,'LineWidth',1.7,'Color',colors(1,:)); hold on; semilogy(z/1000,airyAxis,'LineWidth',1.7,'Color',colors(2,:)); semilogy(z/1000,I_top_axis,'LineWidth',1.7,'Color',colors(3,:)); semilogy(z/1000,I_lg_axis+eps,'LineWidth',1.7,'Color',colors(4,:)); semilogy(z/1000,I_hg_axis+eps,'LineWidth',1.7,'Color',colors(5,:)); title('On-axis irradiance'); xlabel('z (km)'); ylabel('I(0,z) (W/m^2)'); grid on; legend(names,'FontSize',8);
nexttile; plot(r*1000,profile0,'LineWidth',1.5); title('Transverse profile at z=0'); xlabel('x or r (mm)'); ylabel('irradiance (W/m^2)'); grid on; legend(names,'FontSize',8);
nexttile; semilogy(r*1000,max(profile100,1e-10),'LineWidth',1.5); title('Transverse profile at z=100 km'); xlabel('x or r (mm)'); ylabel('irradiance (W/m^2)'); grid on; legend(names,'FontSize',8);
nexttile; axis off; text(0,0.95,sprintf('lambda = 1064 nm\nP = 1 W\nw0 = 10 mm\nBessel cone = 0.05 mrad\nBessel aperture = 0.20 m\nAiry x0 = 50 mm\nTop-hat radius = 10 mm\nRange = 0-100 km\n\nLG01 and HG10 are dark on axis.'),'VerticalAlignment','top','FontSize',12);
sgtitle('Non-Gaussian paraxial beam comparison'); saveas(gcf,fullfile(outdir,'other_beams_comparison_matlab.png'));

%% Propagation maps
zmap=linspace(0,100e3,500); xmap=linspace(-0.15,0.15,420); [X,Z]=meshgrid(xmap,zmap);
Wmap=w0*sqrt(1+(Z/zr).^2); U=sqrt(2)*X./Wmap; fieldLG=4*P./(pi*Wmap.^2).*U.^2.*exp(-U.^2); fieldHG=2*(2*P./(pi*Wmap.^2)).*U.^2.*exp(-U.^2);
fieldB=2*P/(pi*besselCore^2).*besselj(0,kr*X).^2; s=X/airyX0; xx=Z/airyZ0; fieldA=P/(sqrt(2*pi)*airyX0).*abs(airy(s-xx.^2/4+1i*airyA*xx)).^2.*exp(2*airyA*s-airyA*xx.^2);
fieldT=P/(pi*tophatR^2).*ones(size(X));
figure('Color','w','Position',[100 100 1300 1000]); tiledlayout(3,2,'Padding','compact'); fields={fieldB,fieldA,fieldT,fieldLG,fieldHG};
for q=1:5; nexttile; imagesc(zmap/1000,xmap*1000,log10(max(fields{q},1e-10))); axis xy; colorbar; title(names{q}); xlabel('z (km)'); ylabel('transverse (mm)'); end
nexttile; axis off; sgtitle('Beam propagation cross-sections (log10 irradiance)'); saveas(gcf,fullfile(outdir,'other_beams_propagation_maps_matlab.png'));
