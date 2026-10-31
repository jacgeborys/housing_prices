"""
Context layers for the Lublin slides, with Warsaw's fetchers imported (not edited) and pointed at Lublin.
Run from the lublin/ folder:  python fetch_context.py [pois] [noise]

    pois   -> data/osm_pois.gpkg (fetch_pois.py, amenity query only: food & drink, schools, health, culture; Overpass, polite)
    noise  -> data/noise_{road,rail,air}_lden.tif (fetch_noise.py: EEA strategic noise maps, round 4 / 2022, 10 m bands)
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LUBLIN_LONLAT = (22.43, 51.13, 22.71, 51.31)


def main(which):
    if "pois" in which:
        import fetch_pois
        fetch_pois.AREA = 'area["name"="Lublin"]["boundary"="administrative"]["admin_level"="7"]->.a;'
        fetch_pois.PARTS = {"amenity": fetch_pois.PARTS["amenity"]}      # the slides use food & drink only; shop/leisure skipped
        fetch_pois.main()
    if "noise" in which:
        import fetch_noise
        fetch_noise.WARSAW_LONLAT = LUBLIN_LONLAT
        fetch_noise.main()


if __name__ == "__main__":
    main(sys.argv[1:] or ["pois", "noise"])
