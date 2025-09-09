import holoviews as hv
import numpy as np


def plot_func(ids_dict):
    OPTIONS = hv.opts.Curve(framewise=True, responsive=True)

    if not ids_dict:
        curve = curve2 = curve3 = hv.Curve(([], [])).opts(OPTIONS)
    else:
        eq = ids_dict["equilibrium"]
        ts = eq.time_slice[-1]
        f_df_dpsi = ts.profiles_1d.f_df_dpsi
        psi = ts.profiles_1d.psi
        curve = hv.Curve((psi, f_df_dpsi), "Psi", "ff'").opts(OPTIONS)

        ip = np.array([ts.global_quantities.ip for ts in eq.time_slice])
        time = eq.time
        curve2 = hv.Curve((time, ip), "time", "Ip").opts(OPTIONS)

        pfa = ids_dict["pf_active"]
        curr = pfa.coil[0].current.data
        time = pfa.time
        curve3 = hv.Curve((time, curr), "time", "coil current").opts(OPTIONS)
    return curve3
