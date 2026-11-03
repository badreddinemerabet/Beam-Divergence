"""Free-space Gaussian-beam divergence simulation from 0 to 100 km.

Run from this folder:
    python gaussian_beam_sim.py

Baseline inputs are theoretical and editable in make_beams():
- wavelengths: 1064 nm, 1550 nm, 3.5 um, 10.6 um
- waist radius w0: 10 mm (1/e^2 intensity radius)
- power: 1 W per beam
- beam quality M^2: 1 (diffraction-limited)

Equations:
    z_R = pi*w0^2/(M^2*lambda)
    w(z) = w0*sqrt(1 + (z/z_R)^2)
    theta = M^2*lambda/(pi*w0)          # far-field half-angle [rad]
    I(r,z) = 2*P/(pi*w(z)^2)*exp(-2*r^2/w(z)^2)
    |E| = sqrt(2 I/(c eps0))            # peak electric-field amplitude
    |H| = |E|/eta0                      # peak magnetic-field amplitude
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
MU0 = 4.0e-7 * np.pi
ETA0 = np.sqrt(MU0 / EPS0)


@dataclass(frozen=True)
class Beam:
    name: str
    wavelength_nm: float
    waist_mm: float
    power_w: float
    m2: float = 1.0
    color: str = "C0"

    @property
    def wavelength_m(self):
        return self.wavelength_nm * 1e-9

    @property
    def waist_m(self):
        return self.waist_mm * 1e-3

    @property
    def rayleigh_range_m(self):
        return np.pi * self.waist_m**2 / (self.m2 * self.wavelength_m)

    @property
    def divergence_half_angle_rad(self):
        return self.m2 * self.wavelength_m / (np.pi * self.waist_m)

    @property
    def divergence_half_angle_mrad(self):
        return 1e3 * self.divergence_half_angle_rad

    @property
    def divergence_full_angle_mrad(self):
        return 2e3 * self.divergence_half_angle_rad

    def radius(self, z_m):
        z_m = np.asarray(z_m, dtype=float)
        return self.waist_m * np.sqrt(1.0 + (z_m / self.rayleigh_range_m) ** 2)

    def irradiance(self, r_m, z_m):
        w = self.radius(z_m)
        return 2.0 * self.power_w / (np.pi * w**2) * np.exp(-2.0 * np.asarray(r_m) ** 2 / w**2)

    def electric_field_amplitude(self, r_m, z_m):
        return np.sqrt(2.0 * self.irradiance(r_m, z_m) / (C0 * EPS0))

    def magnetic_field_amplitude(self, r_m, z_m):
        return self.electric_field_amplitude(r_m, z_m) / ETA0


def make_beams():
    # Edit this list to study another wavelength, power, waist radius, or M^2.
    return [
        Beam("Near-IR 1064 nm", 1064.0, 10.0, 1.0, 1.0, "#d97706"),
        Beam("Near-IR 1550 nm", 1550.0, 10.0, 1.0, 1.0, "#eab308"),
        Beam("Mid-IR 3.5 um", 3500.0, 10.0, 1.0, 1.0, "#0891b2"),
        Beam("Mid-IR 10.6 um", 10600.0, 10.0, 1.0, 1.0, "#dc2626"),
    ]


def print_summary(beams):
    print("\nFree-space Gaussian-beam summary")
    print("w0 = 10 mm is the 1/e^2 intensity radius; P = 1 W; M^2 = 1\n")
    header = (f"{'Beam':<20} {'lambda':>10} {'zR (m)':>12} "
              f"{'half-angle (mrad)':>19} {'full-angle (mrad)':>19} {'w(100 km) (m)':>16}")
    print(header)
    print("-" * len(header))
    for b in beams:
        print(f"{b.name:<20} {b.wavelength_nm:10.1f} {b.rayleigh_range_m:12.3f} "
              f"{b.divergence_half_angle_mrad:19.5f} {b.divergence_full_angle_mrad:19.5f} "
              f"{b.radius(100_000):16.3f}")


def export_tables(beams, output_dir):
    z = np.linspace(0.0, 100_000.0, 1001)
    with (output_dir / "beam_propagation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beam", "wavelength_nm", "z_m", "radius_m", "on_axis_irradiance_W_m2",
                         "on_axis_E_V_m", "on_axis_H_A_m", "total_power_W"])
        for b in beams:
            for zi in z:
                writer.writerow([b.name, b.wavelength_nm, zi, b.radius(zi), b.irradiance(0, zi),
                                 b.electric_field_amplitude(0, zi), b.magnetic_field_amplitude(0, zi), b.power_w])

    with (output_dir / "beam_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beam", "wavelength_nm", "waist_radius_mm", "power_W", "M2",
                         "rayleigh_range_m", "half_angle_rad", "half_angle_mrad",
                         "full_angle_mrad", "radius_at_100km_m"])
        for b in beams:
            writer.writerow([b.name, b.wavelength_nm, b.waist_mm, b.power_w, b.m2, b.rayleigh_range_m,
                             b.divergence_half_angle_rad, b.divergence_half_angle_mrad,
                             b.divergence_full_angle_mrad, b.radius(100_000)])


def create_plots(beams, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    z = np.linspace(0, 100_000, 800)
    r = np.linspace(0, 0.5, 600)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for b in beams:
        ax[0, 0].plot(z / 1000, b.radius(z), label=b.name, color=b.color, lw=2)
        ax[0, 1].plot(z / 1000, b.irradiance(0, z), label=b.name, color=b.color, lw=2)
        ax[1, 0].plot(r * 1000, b.irradiance(r, 0), label=b.name, color=b.color, lw=2)
        ax[1, 1].plot(z / 1000, b.electric_field_amplitude(0, z), label=b.name, color=b.color, lw=2)
    ax[0, 0].set(title="1/e² beam radius", xlabel="Propagation distance z (km)", ylabel="w(z) (m)")
    ax[0, 1].set(title="On-axis irradiance", xlabel="Propagation distance z (km)", ylabel="I(0,z) (W/m²)", yscale="log")
    ax[1, 0].set(title="Transverse irradiance at waist", xlabel="Radius r (mm)", ylabel="I(r,0) (W/m²)")
    ax[1, 1].set(title="On-axis electric-field amplitude", xlabel="Propagation distance z (km)", ylabel="|E(0,z)| (V/m)")
    for a in ax.ravel():
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Free-space wavelength-dependent Gaussian beams: 0–100 km", fontsize=15)
    fig.savefig(output_dir / "beam_comparison.png", dpi=180)
    plt.close(fig)

    # 2-D axisymmetric cross-sections. The transverse coordinate is displayed in mm.
    z_map = np.linspace(0, 100_000, 500)
    y_map = np.linspace(-120, 120, 500)  # metres; enough for the 10.6 um beam at 100 km
    Z, Y = np.meshgrid(z_map, y_map)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True, sharex=True, sharey=True)
    for a, b in zip(axes.ravel(), beams):
        intensity = b.irradiance(Y, Z)
        # Log scale keeps both the narrow waist and the weak 100 km tail visible.
        img = a.pcolormesh(Z / 1000, Y, np.maximum(intensity, 1e-12), shading="auto",
                           cmap="magma", norm=LogNorm(vmin=1e-8, vmax=max(1.0, intensity.max())))
        a.set_title(b.name)
        a.set_xlabel("z (km)")
        a.set_ylabel("transverse y (m)")
        a.grid(alpha=0.15)
        fig.colorbar(img, ax=a, label="Irradiance (W/m², log scale)")
    fig.suptitle("2-D irradiance cross-sections (log scale)", fontsize=15)
    fig.savefig(output_dir / "beam_intensity_maps.png", dpi=180)
    plt.close(fig)


def main():
    beams = make_beams()
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print_summary(beams)
    export_tables(beams, output_dir)
    create_plots(beams, output_dir)
    print(f"\nSaved plots and tables to: {output_dir}")


if __name__ == "__main__":
    main()
