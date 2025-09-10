import holoviews as hv
import numpy as np

OPTIONS = hv.opts.Curve(framewise=True, responsive=True)


def plot1(ids_dict):
    if not ids_dict:
        psi = f_df_dpsi = []
    else:
        eq = ids_dict["equilibrium"]
        ts = eq.time_slice[-1]
        f_df_dpsi = ts.profiles_1d.f_df_dpsi
        psi = ts.profiles_1d.psi
    curve = hv.Curve((psi, f_df_dpsi), "Psi", "ff'").opts(OPTIONS)
    return curve


def plot2(ids_dict):
    if not ids_dict:
        time = ip = []
    else:
        eq = ids_dict["equilibrium"]
        ip = np.array([ts.global_quantities.ip for ts in eq.time_slice])
        time = eq.time
    curve = hv.Curve((time, ip), "time", "Ip").opts(OPTIONS)
    return curve


def plot3(ids_dict):
    if not ids_dict:
        time = curr = []
    else:
        pfa = ids_dict["pf_active"]
        curr = pfa.coil[0].current.data
        time = pfa.time
    curve = hv.Curve((time, curr), "time", "coil current").opts(OPTIONS)
    return curve


def plot4(ids_dict):
    if not ids_dict or "equilibrium" not in ids_dict:
        empty_coords = np.array([0, 1])
        empty_x, empty_y = np.meshgrid(empty_coords, empty_coords)
        empty_data = np.zeros((2, 2))
        return hv.QuadMesh((empty_x, empty_y, empty_data)).opts(
            framewise=True, responsive=True, colorbar=True, title="Waiting for data..."
        )

    eq = ids_dict["equilibrium"]
    ts = eq.time_slice[-1]
    profiles_2d = ts.profiles_2d[0]

    r = np.array(profiles_2d.r)
    z = np.array(profiles_2d.z)
    psi = np.array(profiles_2d.psi)

    if r.shape != psi.shape:
        r_1d = r[0, :]
        z_1d = z[:, 0]
        r, z = np.meshgrid(r_1d, z_1d)

    img = hv.QuadMesh((r, z, psi), kdims=["r", "z"], vdims=["psi"])

    return img.opts(
        framewise=True,
        responsive=True,
        colorbar=True,
        title=f"Time: {eq.time[-1]:.4f}s",
    )
