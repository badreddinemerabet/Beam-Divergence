"""Generate corrected z=0 irradiance maps for all comparison beams."""
from pathlib import Path
from other_beams_sim import plot_z0

if __name__ == '__main__':
    outdir = Path(__file__).resolve().parent / 'outputs_other_beams'
    outdir.mkdir(exist_ok=True)
    plot_z0(outdir)
    print(f'Wrote corrected z=0 figures to {outdir}')
