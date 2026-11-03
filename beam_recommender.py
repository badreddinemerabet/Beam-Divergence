"""Physics-generated neural-network recommender for minimum beam divergence.

The model learns a regression mapping from beam/propagation parameters to an
estimated effective half-divergence. At inference it evaluates every candidate
beam and returns the lowest-divergence recommendation.

This is a ranking aid, not a replacement for a full diffraction solver.
"""
from pathlib import Path
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

SEED=7
BEAMS=['Gaussian','Bessel','Airy','Top-hat','LG01','HG10','Bessel-Gaussian']
BEAM_INDEX={b:i for i,b in enumerate(BEAMS)}


def physics_divergence(beam, wavelength_m, waist_m, m2, z_m, receiver_aperture_m, bessel_cone_rad, bessel_aperture_m, airy_scale_m):
    """Effective half-divergence in rad, defined as characteristic width / z."""
    z=max(float(z_m),1e-6); lam=wavelength_m; k=2*np.pi/lam
    if beam in ('Gaussian','LG01','HG10'):
        mode_factor={'Gaussian':1.0,'LG01':2.0,'HG10':2.0}[beam]
        zr=np.pi*waist_m**2/(m2*mode_factor*lam)
        width=waist_m*np.sqrt(1+(z/zr)**2)
        return width/z
    if beam=='Top-hat':
        diameter=max(2*waist_m,1e-9)
        return 1.22*lam/diameter
    if beam=='Airy':
        # Effective main-lobe width for a finite-energy paraxial Airy beam.
        return np.sqrt(airy_scale_m**2+(lam*z/(np.pi*airy_scale_m))**2)/z
    if beam=='Bessel':
        kr=k*np.sin(max(bessel_cone_rad,1e-12)); core=2.405/kr
        z_nd=max(bessel_aperture_m/np.tan(max(bessel_cone_rad,1e-12)),1e-6)
        if z <= z_nd:
            return np.sqrt(core**2+(bessel_cone_rad*z)**2)/z
        # after the finite aperture's nondiffracting range, use its geometric envelope
        return np.sqrt(core**2+(bessel_cone_rad*z)**2+(waist_m*z/z_nd)**2)/z
    if beam=='Bessel-Gaussian':
        kr=k*np.sin(max(bessel_cone_rad,1e-12)); core=2.405/kr
        env0=max(3*waist_m,1e-6); zr=np.pi*env0**2/lam
        width=np.sqrt(core**2+(env0*np.sqrt(1+(z/zr)**2))**2)
        return width/z
    raise ValueError(beam)


def make_features(beam, scenario):
    lam,w0,m2,z,ap,cone,bap,airy=scenario
    onehot=np.zeros(len(BEAMS)); onehot[BEAM_INDEX[beam]]=1
    # Log scales make the wide wavelength/distance ranges learnable.
    return np.r_[onehot, np.log10(lam*1e9), np.log10(w0*1e3), m2,
                 np.log10(z/1000), np.log10(ap), np.log10(cone*1e3),
                 np.log10(bap), np.log10(airy*1e3)]


def random_scenarios(n, rng):
    # wavelength: 0.4–12 um; waist: 1–100 mm; distance: 0.1–100 km.
    return list(zip(10**rng.uniform(np.log10(0.4e-6),np.log10(12e-6),n),
                    10**rng.uniform(np.log10(1e-3),np.log10(0.1),n),
                    rng.uniform(1,2.5,n),
                    10**rng.uniform(np.log10(100),np.log10(100e3),n),
                    10**rng.uniform(np.log10(0.02),np.log10(5.0),n),
                    10**rng.uniform(np.log10(0.01e-3),np.log10(0.3e-3),n),
                    10**rng.uniform(np.log10(0.03),np.log10(2.0),n),
                    10**rng.uniform(np.log10(2e-3),np.log10(0.15),n)))


class MLP:
    def __init__(self,nin,hidden=(40,24),seed=SEED):
        rng=np.random.default_rng(seed); h1,h2=hidden
        self.W1=rng.normal(0,np.sqrt(2/nin),(nin,h1)); self.b1=np.zeros(h1)
        self.W2=rng.normal(0,np.sqrt(2/h1),(h1,h2)); self.b2=np.zeros(h2)
        self.W3=rng.normal(0,np.sqrt(2/h2),(h2,1)); self.b3=np.zeros(1)
    def forward(self,X):
        self.z1=X@self.W1+self.b1; self.a1=np.maximum(self.z1,0)
        self.z2=self.a1@self.W2+self.b2; self.a2=np.maximum(self.z2,0)
        return self.a2@self.W3+self.b3
    def predict(self,X): return self.forward(X).ravel()
    def fit(self,X,y,epochs=900,lr=0.003):
        n=len(X)
        for epoch in range(epochs):
            pred=self.forward(X).ravel(); err=pred-y; loss=float(np.mean(err**2))
            d=2*err[:,None]/n
            gW3=self.a2.T@d; gb3=d.sum(0); da2=d@self.W3.T; dz2=da2*(self.z2>0)
            gW2=self.a1.T@dz2; gb2=dz2.sum(0); da1=dz2@self.W2.T; dz1=da1*(self.z1>0)
            gW1=X.T@dz1; gb1=dz1.sum(0)
            for p,g in [(self.W1,gW1),(self.b1,gb1),(self.W2,gW2),(self.b2,gb2),(self.W3,gW3),(self.b3,gb3)]: p-=lr*g
            if epoch%150==0: print(f'epoch {epoch:4d}  log-divergence MSE={loss:.6f}')


def train_and_evaluate(outdir):
    rng=np.random.default_rng(SEED); scenarios=random_scenarios(3500,rng)
    X=[]; y=[]; labels=[]
    for s in scenarios:
        for beam in BEAMS:
            X.append(make_features(beam,s)); y.append(np.log10(physics_divergence(beam,*s)+1e-30)); labels.append(beam)
    X=np.asarray(X); y=np.asarray(y)
    idx=rng.permutation(len(X)); split=int(0.8*len(X)); tr,te=idx[:split],idx[split:]
    mu=X[tr].mean(0); sig=X[tr].std(0)+1e-12; Xn=(X-mu)/sig
    ymu=float(y[tr].mean()); ysig=float(y[tr].std()+1e-12)
    model=MLP(X.shape[1]); model.fit(Xn[tr],(y[tr]-ymu)/ysig,epochs=1400,lr=0.003); pred=model.predict(Xn[te])*ysig+ymu
    rmse=np.sqrt(np.mean((pred-y[te])**2)); print(f'test RMSE in log10(rad): {rmse:.4f}')
    # Ranking accuracy on held-out complete scenarios, not randomly shuffled rows.
    correct=0; total=0; test_scenarios=random_scenarios(500,rng)
    for s in test_scenarios:
        rows=np.asarray([make_features(b,s) for b in BEAMS]); actual=np.asarray([np.log10(physics_divergence(b,*s)+1e-30) for b in BEAMS])
        p=model.predict((rows-mu)/sig)*ysig+ymu
        if np.argmin(p)==np.argmin(actual): correct+=1
        total+=1
    print(f'beam-selection accuracy on grouped test rows: {correct/max(total,1):.3f}')
    with open(outdir/'beam_recommender_model.pkl','wb') as f: pickle.dump({'model':model,'mu':mu,'sig':sig,'ymu':ymu,'ysig':ysig,'beams':BEAMS},f)
    with open(outdir/'beam_recommender_metrics.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['metric','value']); w.writerow(['test_log10_divergence_RMSE',rmse]); w.writerow(['selection_accuracy',correct/max(total,1)]); w.writerow(['training_rows',len(tr)])
    # Diagnostic plot.
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(y[te],pred,s=3,alpha=.25); lo=min(y[te].min(),pred.min()); hi=max(y[te].max(),pred.max()); ax.plot([lo,hi],[lo,hi],'k--'); ax.set(xlabel='physics log10(effective divergence rad)',ylabel='MLP prediction',title='Neural-network validation'); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(outdir/'beam_recommender_validation.png',dpi=180); plt.close(fig)
    return model,mu,sig,ymu,ysig


def recommend(model,mu,sig,ymu,ysig,scenario):
    rows=np.asarray([make_features(b,scenario) for b in BEAMS]); pred=model.predict((rows-mu)/sig)*ysig+ymu; div=np.array([physics_divergence(b,*scenario) for b in BEAMS]);
    return [(BEAMS[i],div[i],10**pred[i]) for i in np.argsort(pred)]


def main():
    outdir=Path(__file__).resolve().parent/'outputs_beam_recommender'; outdir.mkdir(exist_ok=True)
    model,mu,sig,ymu,ysig=train_and_evaluate(outdir)
    # Example requested-style case: 1064 nm, 10 mm waist, M2=1, 100 km.
    scenario=(1064e-9,10e-3,1.0,100e3,0.20,0.05e-3,0.20,0.05)
    ranked=recommend(model,mu,sig,ymu,ysig,scenario)
    print('\nExample recommendation: 1064 nm, w0=10 mm, M2=1, z=100 km')
    for rank,(name,phys,ml) in enumerate(ranked,1): print(f'{rank}. {name:18s} physics={phys:.4e} rad, ML={ml:.4e} rad')
    with open(outdir/'example_recommendation.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['rank','beam','physics_effective_half_divergence_rad','mlp_predicted_half_divergence_rad']);
        for rank,(name,phys,ml) in enumerate(ranked,1): w.writerow([rank,name,phys,ml])

if __name__=='__main__': main()
