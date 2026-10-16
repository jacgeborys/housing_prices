"""
Shared RCN cleaning for every pipeline (build_viz.py, analyze_rcn.py -> price_index.py).

residential_clean(rcn): rcn = the raw lokale layer with lower-case columns and numeric lok_cena_brutto.
Returns residential flats only, without records whose price doesn't belong to one whole flat:
  - sales of a share of a flat (nier_udzial 1/2, 1/3 ...): the price covers a fraction, the area the whole flat
  - flats in a multi-flat deed that carry the deed's total price (tran_cena_brutto) as their own
A single flat sold with a garage and priced as a whole is kept (inflated only by a parking space).
"""
import pandas as pd


def residential_clean(rcn, verbose=True):
    deed = rcn["tran_lokalny_id_iip"]
    n_units = deed.map(deed.value_counts())
    is_flat = rcn["lok_funkcja"] == "mieszkalna"
    n_flats = deed.map(deed[is_flat].value_counts()).fillna(0)
    tran_price = pd.to_numeric(rcn["tran_cena_brutto"], errors="coerce")
    res = rcn[is_flat].copy()
    n_units, n_flats, tran_price = n_units[is_flat], n_flats[is_flat], tran_price[is_flat]

    share = res["nier_udzial"].fillna("").astype(str).str.strip()
    frac = share.str.extract(r"^(\d+)/(\d+)$").astype(float)
    partial = frac[0].notna() & (frac[0] != frac[1])
    total_copied = res["lok_cena_brutto"].eq(tran_price) & (n_flats > 1)
    with_garage = res["lok_cena_brutto"].eq(tran_price) & (n_flats == 1) & (n_units > 1)
    if verbose:
        print(f"  Residential: {len(res):,}; dropped {int(partial.sum()):,} partial-share sales and "
              f"{int(total_copied.sum()):,} flats carrying a multi-flat deed's total price; kept "
              f"{int(with_garage.sum()):,} flat+garage deeds priced as a whole")
    return res[~partial & ~total_copied].copy()
