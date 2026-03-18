"""
Transmission spectra near 1550 nm for optimized MMI splitter designs.

Uses the standard MEEP two-run normalization:
  1) Reference run with straight waveguide only -> records input flux
  2) Design run with optimized geometry -> records output flux
  T = output_flux / reference_flux

Usage:
    python transmission_spectra.py                          # run all opt files
    python transmission_spectra.py optresult(11).npz        # run a specific file
    python transmission_spectra.py optresult(6).npz optresult(11).npz  # multiple
"""

import meep as mp
import meep.adjoint as mpa
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

mp.verbosity(0)

# ── Materials ──
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

# ── Wavelength sweep ──
WAVELENGTHS =  np.linspace(1.35, 1.75, 150)  # µm, broad range centered on 1550 nm
FREQUENCIES = 1.0 / WAVELENGTHS

# ── Default geometry parameters (for files without Parameter key) ──
DEFAULTS = dict(
    waveguide_width=0.5,
    design_region_width=2.5,
    design_region_height=2.5,
    arm_separation=1.0,
    waveguide_length=0.5,
    pml_size=1.0,
    resolution=30,
)


def load_design(filepath):
    """Load an npz file and return geometry weights + simulation parameters."""
    data = np.load(filepath, allow_pickle=True)
    geo = data["Geo"]
    msg = str(data.get("Message", ""))
    beta = float(data["Beta"]) if "Beta" in data else 64.0
    eta_i = float(data["eta_i"]) if "eta_i" in data else 0.5

    if "Parameter" in data:
        p = data["Parameter"]
        params = dict(
            waveguide_width=float(p[0]),
            design_region_width=float(p[1]),
            design_region_height=float(p[2]),
            arm_separation=float(p[3]),
            waveguide_length=float(p[4]),
            pml_size=float(p[5]),
            resolution=int(p[6]),
        )
    else:
        params = DEFAULTS.copy()

    return geo, params, beta, eta_i, msg


def apply_mapping(geo_weights, params, beta, eta_i):
    """Apply the conic filter + tanh projection mapping (same as notebook)."""
    drw = params["design_region_width"]
    drh = params["design_region_height"]
    res = params["resolution"]
    design_region_resolution = int(5 * res)

    minimum_length = 0.09
    eta_e = 0.55
    filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)

    filtered = mpa.conic_filter(
        geo_weights, filter_radius, drw, drh, design_region_resolution,
    )
    projected = mpa.tanh_projection(filtered, beta, eta_i)
    return projected.flatten()


def make_sim(params, design_weights=None):
    """
    Build a MEEP simulation.
    If design_weights is None, builds a reference sim (straight waveguide only).
    If provided, includes the design region with those weights.
    """
    ww = params["waveguide_width"]
    drw = params["design_region_width"]
    drh = params["design_region_height"]
    arm_sep = params["arm_separation"]
    wl = params["waveguide_length"]
    pml = params["pml_size"]
    res = params["resolution"]

    design_region_resolution = int(5 * res)
    Nx = int(design_region_resolution * drw) + 1
    Ny = int(design_region_resolution * drh) + 1

    Sx = 2 * pml + 2 * wl + drw
    Sy = 2 * pml + drh + 0.5
    cell_size = mp.Vector3(Sx, Sy)
    pml_layers = [mp.PML(pml)]

    # Base geometry: input waveguide + two output arms
    geometry = [
        mp.Block(center=mp.Vector3(x=-Sx / 4), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, ww, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=arm_sep / 2), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, ww, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=-arm_sep / 2), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, ww, 0)),
    ]

    # Add design region if weights are provided
    if design_weights is not None:
        design_variables = mp.MaterialGrid(
            mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN"
        )
        design_variables.weights = np.array(design_weights).reshape((Nx, Ny))
        geometry.append(
            mp.Block(center=mp.Vector3(), size=mp.Vector3(drw, drh, 0),
                     material=design_variables)
        )

    # Source (matches notebook: NO_DIRECTION + eig_kpoint for +x propagation)
    fcen = 1 / 1.55
    fwidth = 0.3 * fcen
    source_center = mp.Vector3(-Sx / 2 + pml + wl / 3, 0, 0)
    source_size = mp.Vector3(0, 2, 0)
    kpoint = mp.Vector3(1, 0, 0)
    sources = [
        mp.EigenModeSource(
            mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            eig_band=1, direction=mp.NO_DIRECTION, eig_kpoint=kpoint,
            size=source_size, center=source_center,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=SiO2,
        resolution=res,
    )

    return sim, Sx, Sy


def add_flux_monitors(sim, params, frequencies):
    """Add flux monitors at the input and both output arms. Returns monitor objects."""
    wl = params["waveguide_length"]
    pml = params["pml_size"]
    arm_sep = params["arm_separation"]

    Sx = 2 * pml + 2 * wl + params["design_region_width"]

    # Input monitor: between source and design region
    input_fr = mp.FluxRegion(
        center=mp.Vector3(x=-Sx / 2 + pml + 2 * wl / 3),
        size=mp.Vector3(y=1.5),
        direction=mp.X,
    )
    # Output monitors: between design region and PML
    top_fr = mp.FluxRegion(
        center=mp.Vector3(Sx / 2 - pml - 2 * wl / 3, arm_sep / 2, 0),
        size=mp.Vector3(y=arm_sep),
        direction=mp.X,
    )
    bot_fr = mp.FluxRegion(
        center=mp.Vector3(Sx / 2 - pml - 2 * wl / 3, -arm_sep / 2, 0),
        size=mp.Vector3(y=arm_sep),
        direction=mp.X,
    )

    input_mon = sim.add_flux(frequencies, input_fr)
    top_mon = sim.add_flux(frequencies, top_fr)
    bot_mon = sim.add_flux(frequencies, bot_fr)

    return input_mon, top_mon, bot_mon


def run_transmission(params, design_weights, frequencies):
    """
    Single-run measurement: input flux and output flux from the same simulation.
    T = output_flux / input_flux (net Poynting flux accounts for reflections).
    """
    print("  Running simulation...")
    sim, _, _ = make_sim(params, design_weights=design_weights)
    input_mon, top_mon, bot_mon = add_flux_monitors(sim, params, frequencies)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(), 1e-7))

    input_flux = np.array(mp.get_fluxes(input_mon))
    top_flux = np.array(mp.get_fluxes(top_mon))
    bot_flux = np.array(mp.get_fluxes(bot_mon))
    sim.reset_meep()

    T_top = top_flux / input_flux
    T_bot = bot_flux / input_flux
    T_total = T_top + T_bot

    return T_top, T_bot, T_total


def plot_spectra(wavelengths, results, save_dir="spectra"):
    """Plot and save transmission spectra for all designs."""
    os.makedirs(save_dir, exist_ok=True)

    for name, (T_top, T_bot, T_total, msg) in results.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wavelengths * 1e3, T_top, label="Top arm", linewidth=1.5)
        ax.plot(wavelengths * 1e3, T_bot, label="Bottom arm", linewidth=1.5)
        ax.plot(wavelengths * 1e3, T_total, "k--", label="Total", linewidth=1.5)
        ax.axvline(1550, color="gray", linestyle=":", alpha=0.5, label="1550 nm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission")
        ax.set_title("")
        ax.set_xlim(wavelengths[0] * 1e3, wavelengths[-1] * 1e3)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = os.path.join(save_dir, f"{name}_spectra.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  Saved: {outpath}")

    # Combined plot
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, (T_top, T_bot, T_total, msg) in results.items():
            ax.plot(wavelengths * 1e3, T_total, label=f"{name} (total)")
        ax.axvline(1550, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Total Transmission")
        ax.set_title("Combined Transmission Spectra")
        ax.set_xlim(wavelengths[0] * 1e3, wavelengths[-1] * 1e3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = os.path.join(save_dir, "combined_spectra.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  Saved: {outpath}")


def main():
    opts_dir = "./opts"

    if len(sys.argv) > 1:
        files = [os.path.join(opts_dir, f) for f in sys.argv[1:]]
    else:
        files = sorted(
            [os.path.join(opts_dir, f) for f in os.listdir(opts_dir) if f.endswith(".npz")]
        )

    wavelengths = WAVELENGTHS
    frequencies = FREQUENCIES
    results = {}

    for filepath in files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        geo, params, beta, eta_i, msg = load_design(filepath)
        print(f"  Message: {msg}")
        print(f"  Resolution: {params['resolution']}, Beta: {beta:.1f}")

        design_weights = apply_mapping(geo, params, beta, eta_i)

        T_top, T_bot, T_total = run_transmission(params, design_weights, frequencies)

        results[name] = (T_top, T_bot, T_total, msg)

        idx_1550 = np.argmin(np.abs(wavelengths - 1.55))
        print(f"  @ 1550 nm: Top={T_top[idx_1550]:.4f}, Bot={T_bot[idx_1550]:.4f}, "
              f"Total={T_total[idx_1550]:.4f}")

    plot_spectra(wavelengths, results)

    # Save numerical data
    os.makedirs("spectra", exist_ok=True)
    np.savez(
        "spectra/transmission_data.npz",
        wavelengths=wavelengths,
        **{f"{k}_top": v[0] for k, v in results.items()},
        **{f"{k}_bot": v[1] for k, v in results.items()},
        **{f"{k}_total": v[2] for k, v in results.items()},
    )
    print("\nNumerical data saved to spectra/transmission_data.npz")


if __name__ == "__main__":
    main()
