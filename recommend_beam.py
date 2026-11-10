"""Use the trained beam recommender for a new propagation case.

Example:
  python recommend_beam.py --wavelength-nm 1064 --waist-mm 10 --m2 1 --distance-km 100
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import beam_recommender as br
from beam_recommender import BEAMS, make_features, physics_divergence, recommend
import __main__
__main__.MLP = br.MLP

p=argparse.ArgumentParser()
p.add_argument('--wavelength-nm',type=float,default=1064)
p.add_argument('--waist-mm',type=float,default=10)
p.add_argument('--m2',type=float,default=1.0)
p.add_argument('--distance-km',type=float,default=100)
p.add_argument('--receiver-aperture-m',type=float,default=0.20)
p.add_argument('--bessel-cone-mrad',type=float,default=0.05)
p.add_argument('--bessel-aperture-m',type=float,default=0.20)
p.add_argument('--airy-scale-mm',type=float,default=50)
a=p.parse_args()
from beam_models import positive
try:
    positive(**vars(a))
    if a.m2<1: raise ValueError('M2 must be at least 1')
except ValueError as exc:
    p.error(str(exc))
root=Path(__file__).resolve().parent
with open(root/'outputs_beam_recommender/beam_recommender_model.pkl','rb') as f: saved=pickle.load(f)
scenario=(a.wavelength_nm*1e-9,a.waist_mm*1e-3,a.m2,a.distance_km*1000,a.receiver_aperture_m,a.bessel_cone_mrad*1e-3,a.bessel_aperture_m,a.airy_scale_mm*1e-3)
ranked=recommend(saved['model'],saved['mu'],saved['sig'],saved['ymu'],saved['ysig'],scenario)
physics_winner=min(ranked,key=lambda x:x[1]); ml_winner=ranked[0]
print(f'Inputs: lambda={a.wavelength_nm:g} nm, w0={a.waist_mm:g} mm, M2={a.m2:g}, z={a.distance_km:g} km')
print(f'Lowest heuristic width-score beam: {physics_winner[0]} ({physics_winner[1]:.5e} rad)')
print(f'MLP predicted ranking winner: {ml_winner[0]} ({ml_winner[2]:.5e} rad predicted)')
print('\nApproximate model ranking; not an engineering beam recommendation:')
for i,(name,phys,ml) in enumerate(ranked,1): print(f'{i}. {name:18s} physics={phys:.5e} rad, MLP={ml:.5e} rad')

