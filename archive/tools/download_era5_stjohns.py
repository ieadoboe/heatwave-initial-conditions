"""
Download ERA5 data for the St. John's airport region (~30km x 30km domain),
derived from the coordinates in convert_erainterim_coord_to_normal.py.

Region:
  Anchor: lat=49.75N, lon=305.5 (=-54.5E)
  16x16 grid points at 0.25-degree resolution → 4°x4° domain
  Bounding box: N=49.75, W=-54.5, S=45.75, E=-50.5

Requirements:
  pip install cdsapi
  Set up ~/.cdsapirc with your CDS API key:
    https://cds.climate.copernicus.eu/api-how-to
"""

import cdsapi

# -- Coordinates from convert_erainterim_coord_to_normal.py --
# Anchor point (St. John's airport area)
anchor_lat = 49.75      # degrees N
anchor_lon_360 = 305.5  # degrees (0-360)
grid_points = 16        # 16x16 domain
res = 0.25              # ERA5 resolution in degrees

# Convert anchor lon to -180/180 convention
anchor_lon = ((anchor_lon_360 + 180) % 360) - 180  # -54.5

# Bounding box: ERA5 latitudes go N→S, so increasing index = decreasing lat
north = anchor_lat
south = anchor_lat - grid_points * res              # 49.75 - 4.0 = 45.75
west = anchor_lon                           # noqa  # -54.5
east = anchor_lon + grid_points * res       # noqa  # -54.5 + 4.0 = -50.5

print(f"Download region: N={north}, W={west}, S={south}, E={east}")

# -- CDS API request --
c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential",
        ],
        "pressure_level": ["1000", "850", "700", "500"],
        "year":  "2017",
        "month": "05",
        "day":   [f"{d:02d}" for d in range(1, 32)],
        "time":  [f"{h:02d}:00" for h in range(0, 24)],
        "area":  [north, west, south, east],   # [N, W, S, E]
        "format": "netcdf",
    },
    "era5_stjohns_may2017_pressure_levels.nc",
)

print("Download complete: era5_stjohns_may2017_pressure_levels.nc")
