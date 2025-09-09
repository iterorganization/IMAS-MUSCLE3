import holoviews as hv
import numpy as np


def plot_func(ids):
    OPTIONS = hv.opts.Curve(framewise=True, responsive=True)
    if not ids:
        curve = curve2 = hv.Curve(([], [])).opts(OPTIONS)
    else:
        ts = ids.time_slice[-1]
        f_df_dpsi = ts.profiles_1d.f_df_dpsi
        psi = ts.profiles_1d.psi
        curve = hv.Curve((psi, f_df_dpsi), "Psi", "ff'").opts(OPTIONS)

        ip = np.array([ts.global_quantities.ip for ts in ids.time_slice])
        time = ids.time
        curve2 = hv.Curve((time, ip), "time", "Ip").opts(OPTIONS)
    return curve
