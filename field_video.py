"""
Animate the Ez field propagating through an optimized MMI splitter design.

Produces a .gif showing the time evolution of the Ez field overlaid on
the device geometry (epsilon).

Usage:
    python field_video.py optresult(11).npz
    python field_video.py optresult(6).npz --fps 15 --duration 80
"""

import meep as mp
import meep.adjoint as mpa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os

mp.verbosity(0)

Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

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


def main():
    parser = argparse.ArgumentParser(description="Animate Ez field through MMI splitter")
    parser.add_argument("design", help="NPZ file from opts/ (e.g. optresult(11).npz)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--duration", type=float, default=80,
                        help="Simulation time in MEEP units (default: 60)")
    parser.add_argument("--dt", type=float, default=0.6,
                        help="Time between frames in MEEP units (default: 0.6)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: <design_name>_field.gif)")
    args = parser.parse_args()

    filepath = args.design
    if not os.path.exists(filepath):
        filepath = os.path.join("opts", args.design)

    geo, params, beta, eta_i, msg = load_design(filepath)
    design_weights = apply_mapping(geo, params, beta, eta_i)

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

    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN"
    )
    design_variables.weights = np.array(design_weights).reshape((Nx, Ny))

    geometry = [
        mp.Block(center=mp.Vector3(x=-Sx / 4), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, ww, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=arm_sep / 2), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, ww, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=-arm_sep / 2), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, ww, 0)),
        mp.Block(center=mp.Vector3(), size=mp.Vector3(drw, drh, 0),
                 material=design_variables),
    ]

    fcen = 1 / 1.55
    fwidth = 0.2 * fcen
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

    # Collect Ez snapshots
    frames = []
    dt = args.dt
    total_time = args.duration

    def capture_frame(sim):
        ez = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
        frames.append(ez.copy().T)  # transpose to (y, x) for imshow

    sim.run(mp.at_every(dt, capture_frame), until=total_time)

    # Get epsilon for overlay
    eps = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
    eps = eps.T

    sim.reset_meep()

    # Build animation
    print(f"Captured {len(frames)} frames, building animation...")
    vmax = max(np.max(np.abs(f)) for f in frames) * 0.7

    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [-Sx / 2, Sx / 2, -Sy / 2, Sy / 2]

    ax.imshow(eps, cmap="binary", extent=extent, origin="lower", alpha=0.3)
    im = ax.imshow(frames[0], cmap="RdBu_r", extent=extent, origin="lower",
                   vmin=-vmax, vmax=vmax, alpha=0.9)
    plt.colorbar(im, ax=ax, label="Ez")
    ax.set_xlabel(r"x ($\mu$m)")
    ax.set_ylabel(r"y ($\mu$m)")
    time_text = ax.set_title(f"t = 0.0")

    def update(i):
        im.set_data(frames[i])
        time_text.set_text(f"t = {i * dt:.1f}")
        return [im, time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // args.fps, blit=True)

    # Save
    name = os.path.splitext(os.path.basename(filepath))[0]
    outfile = args.output or f"{name}_field.gif"
    ani.save(outfile, writer="pillow", fps=args.fps)
    plt.close()
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
