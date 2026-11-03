"""Generate physics-labelled beam recommender data over many distances."""
from pathlib import Path
import csv
import numpy as np
from beam_recommender import BEAMS, physics_divergence

SEED=19

def main():
    rng=np.random.default_rng(SEED)
    outdir=Path(__file__).resolve().parent/'outputs_beam_recommender'; outdir.mkdir(exist_ok=True)
    # Explicit distance anchors ensure the model sees near, intermediate, and long range.
    anchors_km=np.array([0.1,0.2,0.5,1,2,5,10,20,30,50,75,100,125])
    rows=[]
    # 300 randomized cases around each distance anchor.
    for dkm in anchors_km:
        for _ in range(300):
            lam=10**rng.uniform(np.log10(0.4e-6),np.log10(12e-6))
            w0=10**rng.uniform(np.log10(1e-3),np.log10(0.1))
            m2=rng.uniform(1,2.5)
            z=dkm*1000*10**rng.uniform(-0.06,0.06)
            receiver=10**rng.uniform(np.log10(0.02),np.log10(5.0))
            cone=10**rng.uniform(np.log10(0.01e-3),np.log10(0.3e-3))
            bap=10**rng.uniform(np.log10(0.03),np.log10(2.0))
            airy=10**rng.uniform(np.log10(2e-3),np.log10(0.15))
            for beam in BEAMS:
                div=physics_divergence(beam,lam,w0,m2,z,receiver,cone,bap,airy)
                rows.append([beam,lam*1e9,1.0,z/1000,w0*1000,m2,receiver,cone*1000,bap,airy*1000,div,np.log10(div)])
    path=outdir/'beam_training_dataset.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['beam','wavelength_nm','power_W','distance_km','waist_mm','M2','receiver_aperture_m','bessel_cone_mrad','bessel_aperture_m','airy_scale_mm','effective_half_divergence_rad','log10_divergence']); w.writerows(rows)
    print(f'Wrote {len(rows):,} labelled rows to {path}')
    print('Distance anchors (km):', ', '.join(f'{x:g}' for x in anchors_km))

if __name__=='__main__': main()
