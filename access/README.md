# Transit accessibility (subproject)

"How much can I reach by public transport, and when?" for every valued building, door to door (walk + wait +
ride + transfers), instead of straight-line distance to the centre.

## Setup (no system installs)
- `access/.venv`: Python 3.12 made with `uv venv access/.venv --python 3.12`; `uv pip install --python
  access/.venv/Scripts/python.exe r5py geopandas pyogrio matplotlib` (the main project stays on Python 3.8)
- `access/jdk/`: portable Eclipse Temurin 21 JDK (zip from api.adoptium.net), used via JAVA_HOME by `travel_times.py`
- `access/data/mazowieckie-latest.osm.pbf`: Geofabrik extract (OSM walking network)
- data/, .venv/, jdk/ are gitignored. r5py caches built networks in %LOCALAPPDATA%/r5py: clear it after
  changing the GTFS, or it reuses the old network.

## Pipeline
| Script | Python | Output |
|---|---|---|
| `build_inputs.py` | main (3.8) | `data/gtfs_warsaw.zip` (repaired copy of TransitFrequency's merged GTFS: adds the 46 KM/WKD routes its routes.txt lacks), `data/destinations.gpkg` (250 m cells: job equivalents from BDOT10k floor area by function, shop floor area, leisure POIs; malls BDOT files as offices are reclassified via OSM shop=mall), `data/origins.gpkg` |
| `travel_times.py` | access/.venv | `output/access_<scenario>.csv` per building: jobs within 30 / 45 min, gravity jobs (exp(-t/15)), leisure within 20 min, minutes to six hubs |

Scenarios: Wednesday 7.10.2026 07-09, 11-13, 19-21; Saturday 10.10.2026 11-13; median travel time over every
departure minute in the window. Buildings route from their 250 m cell (median building position): 3,263 origins
instead of 20,711 (about 0.7 s per origin: ~40 min per scenario).

Hubs: Centrum (PKiN), Rondo Daszyńskiego, Służewiec, Warszawa Centralna, Arkadia, University of Warsaw.
