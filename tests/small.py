import h5py, numpy as np
f = h5py.File("/lcrc/project/ECP-EZ/public/compression/NYX-Zarija/z42_n512_l10.h5","r")
for n in ["baryon_density","dark_matter_density","temperature","velocity_x","velocity_y","velocity_z"]:
    a = f["/native_fields/"+n][:]
    print(f"{n:20s} range={a.max()-a.min():.6e}")