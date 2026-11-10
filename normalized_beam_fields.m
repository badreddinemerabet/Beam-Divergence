function fields = normalized_beam_fields(X,Y,z,lambda,P,w0)
% Shared finite-power diagnostics; top-hat interpolation is not Fresnel diffraction.
% Bessel-Gaussian is a normalized ansatz, not a finite-aperture propagator.
k=2*pi/lambda; w=w0.*sqrt(1+(z/(pi*w0^2/lambda)).^2);
rho=hypot(X,Y); kr=k*sin(.05e-3); env=.20.*sqrt(1+(z/(pi*.20^2/lambda)).^2);
t=kr^2.*env.^2/4; A=P./(pi*env.^2/2.*besseli(0,t,1));
b=A.*besselj(0,kr*rho).^2.*exp(-2*rho.^2./env.^2);
x0=.05; a=.08; xi=z/(k*x0^2); sx=X/x0; sy=Y/x0;
fx=airy(sx-xi.^2/4+1i*a*xi).*exp(a*sx-a*xi.^2/2+1i*(sx.*xi/2-xi.^3/12+a^2*xi/2));
fy=airy(sy-xi.^2/4+1i*a*xi).*exp(a*sy-a*xi.^2/2+1i*(sy.*xi/2-xi.^3/12+a^2*xi/2));
integral=exp(2*a^3/3)/(2*sqrt(2*pi*a)); ai=P/(x0*integral)^2.*abs(fx.*fy).^2;
ztr=pi*w0^2/lambda; zeff=max(z,ztr); q=k*w0*rho./zeff;
disk=ones(size(q)); mask=abs(q)>1e-12; disk(mask)=2*besselj(1,q(mask))./q(mask);
blend=min(max(z/ztr,0),1);
top=(1-blend).*P/(pi*w0^2).*(rho<=w0)+blend.*P/(pi*w0^2).*(ztr./zeff).^2.*disk.^2;
rho2=2*rho.^2./w.^2; lg=2*P./(pi*w.^2).*rho2.*exp(-rho2);
hg=2*P./(pi*w.^2).*4.*X.^2./w.^2.*exp(-2*rho.^2./w.^2);
fields={b,ai,top,lg,hg};
end
