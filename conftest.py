import imaspy
import numpy
import pytest


@pytest.fixture
def core_profiles():
    cp = imaspy.IDSFactory("3.40.1").core_profiles()
    # Fill some properties:
    cp.ids_properties.homogeneous_time = 0  # INT_0D
    cp.ids_properties.comment = "Comment"  # STR_0D
    cp.ids_properties.provenance.node.resize(1)
    cp.ids_properties.provenance.node[0].path = "profiles_1d"  # STR_0D
    sources = ["First string", "Second string", "Third!"]
    cp.ids_properties.provenance.node[0].sources = sources  # STR_1D
    # Fill some data
    cp.time = [0.0, 1.0, 2.0]
    cp.profiles_1d.resize(len(cp.time))
    for i in range(len(cp.time)):
        cp.profiles_1d[i].time = cp.time[i]
        cp.profiles_1d[i].grid.rho_tor_norm = numpy.linspace(0.0, 1.0, 16)  # FLT_1D
        cp.profiles_1d[i].ion.resize(1)
        cp.profiles_1d[i].ion[0].state.resize(1)
        cp.profiles_1d[i].ion[0].state[0].z_min = 1.0  # FLT_0D
        cp.profiles_1d[i].ion[0].state[0].z_average = 1.25  # FLT_0D
        cp.profiles_1d[i].ion[0].state[0].z_max = 1.5  # FLT_0D
        cp.profiles_1d[i].ion[0].density = numpy.ones(16)
        cp.profiles_1d[i].ion[0].z_ion = 1 + 1e-8
        cp.profiles_1d[i].ion[0].element.resize(1)
        cp.profiles_1d[i].ion[0].element[0].z_n = 2
        cp.profiles_1d[i].ion[0].element[0].atoms_n = 1
        temperature_fit_local = numpy.arange(4, dtype=numpy.int32)
        cp.profiles_1d[i].electrons.temperature_fit.measured = temperature_fit_local
        cp.profiles_1d[i].electrons.temperature_fit.local = temperature_fit_local
        cp.profiles_1d[i].electrons.density = numpy.ones(16)
    return cp
