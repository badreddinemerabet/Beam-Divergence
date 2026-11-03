"""Train the beam recommender from beam_training_dataset.csv."""
from pathlib import Path
import csv, pickle
import numpy as np
import matplotlib.pyplot as plt
from beam_recommender import BEAMS, MLP, make_features, physics_divergence

SEED=23

def main():
    root=Path(__file__).resolve().parent; out=root/'outputs_beam_recommender'; path=out/'beam_training_dataset.csv'; rng=np.random.default_rng(SEED)
    rows=[]
    with path.open() as f:
        reader=csv.DictReader(f)
        for r in reader:
            scenario=(float(r['wavelength_nm'])*1e-9,float(r['waist_mm'])*1e-3,float(r['M2']),float(r['distance_km'])*1000,float(r['receiver_aperture_m']),float(r['bessel_cone_mrad'])*1e-3,float(r['bessel_aperture_m']),float(r['airy_scale_mm'])*1e-3)
            rows.append((r['beam'],scenario,float(r['log10_divergence'])))
    X=np.asarray([make_features(b,s) for b,s,_ in rows]); y=np.asarray([v for _,_,v in rows]); idx=rng.permutation(len(X)); ntr=int(.8*len(X)); tr,te=idx[:ntr],idx[ntr:]
    mu=X[tr].mean(0); sig=X[tr].std(0)+1e-12; ymu=y[tr].mean(); ysig=y[tr].std()+1e-12
    model=MLP(X.shape[1],hidden=(56,32),seed=SEED); model.fit((X[tr]-mu)/sig,(y[tr]-ymu)/ysig,epochs=1800,lr=.003)
    pred=model.predict((X[te]-mu)/sig)*ysig+ymu; rmse=np.sqrt(np.mean((pred-y[te])**2))
    # Selection accuracy by complete seven-row groups in the ordered dataset.
    correct=0; total=0
    for j in range(0,len(rows)-6,7):
        if j+7>len(rows): break
        s=rows[j][1]; actual=np.array([physics_divergence(b,*s) for b in BEAMS]); feats=np.asarray([make_features(b,s) for b in BEAMS]); pp=10**(model.predict((feats-mu)/sig)*ysig+ymu)
        if np.argmin(actual)==np.argmin(pp): correct+=1
        total+=1
    print(f'rows={len(rows):,}; test log10-divergence RMSE={rmse:.4f}; grouped selection accuracy={correct/total:.3f}')
    with (out/'beam_recommender_model.pkl').open('wb') as f: pickle.dump({'model':model,'mu':mu,'sig':sig,'ymu':ymu,'ysig':ysig,'beams':BEAMS},f)
    with (out/'beam_recommender_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['metric','value']); w.writerow(['dataset_rows',len(rows)]); w.writerow(['test_log10_divergence_RMSE',rmse]); w.writerow(['grouped_selection_accuracy',correct/total]); w.writerow(['distance_anchors_km','0.1,0.2,0.5,1,2,5,10,20,30,50,75,100,125'])
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(y[te],pred,s=3,alpha=.25); lo=min(y[te].min(),pred.min()); hi=max(y[te].max(),pred.max()); ax.plot([lo,hi],[lo,hi],'k--'); ax.set(xlabel='physics log10(effective divergence rad)',ylabel='MLP prediction',title='Multi-distance training validation'); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/'beam_recommender_validation.png',dpi=180); plt.close(fig)

if __name__=='__main__': main()
