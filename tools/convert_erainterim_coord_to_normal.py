import numpy as np
import netCDF4 as nc


def convert_erainterim_coord_to_normal():
    """
    Convert ERA-Interim/ERA5 coordinates from
    (0-360) longitude to (-180 to 180).

    long1: longitude varying from -180 to 180 (180W-180E)
    long3: longitude varying from 0 to 360 (all positive)

    Conversion formulas:
      (0-360) -> (-180 to 180): long1 = ((long3 + 180) % 360) - 180
      (-180 to 180) -> (0-360): long3 = long1 % 360
    """
    filename = "era5_daily_pressurelevels_may2017_sphum_1000hpa.nc"

    # choose very small domain 30km * 30km for St. John's airport
    long_nr = 16
    lat_nr = 16

    with nc.Dataset(filename) as ds:
        long3 = ds.variables["longitude"][:]
        lat3 = ds.variables["latitude"][:]

    # Convert longitude from (0-360) to (-180 to 180)
    long1 = ((long3 + 180) % 360) - 180

    conv = np.column_stack([long3, long1])

    m_idx = np.where(lat3 == 49.75)[0]
    k_idx = np.where(long3 == 305.5000)[0]

    m = m_idx[0]
    k = k_idx[0]

    coordlong1 = conv[k: k + long_nr + 1, :]
    coordlong = coordlong1[:, 1]

    coordlat = lat3[m: m + lat_nr + 1]

    print(f"k = {k}")
    print(f"m = {m}")

    return long3, long1, k, m, coordlong, coordlat


if __name__ == "__main__":
    long3, long1, k, m, coordlong, coordlat = convert_erainterim_coord_to_normal()
