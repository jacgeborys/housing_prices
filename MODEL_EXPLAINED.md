# The model, explained simply

*Warsaw flat prices, state of 2026-10-25. Numbers are from the current run. Everything here describes what goes
together with price, not what causes it.*

## In three sentences

1. For every building we work out **what a standard flat there costs**, from 20 years of real sales.
2. We check **which features go with dearer buildings** (age, metro, cafés, noise, trees…), comparing buildings that
   are alike in everything else.
3. What no feature explains, a place being dearer or cheaper than its features suggest, is its **reputation**.

---

## Step 1: one fair price per building

Sales are messy. One flat is big, another small; one is on the ground floor, another on the top; one sold in 2008,
another in 2025. To compare buildings fairly, every sale is converted to the same **"standard flat"**: 2 rooms,
about 50 m², a middle floor, resale, sold in 2025.

How: the model compares sales **inside the same building**. If flats on the top floor of a building usually sell
5 % above those on the 2nd floor, that's the floor effect, learned from about 258,000 sales.

Result: **13,607 buildings** with a standard-flat price (only buildings with at least 3 sales, since one or two
sales are too noisy).

## Step 2: price tags

Now the question: why are some buildings dearer than others? The model gives each feature a **price tag** and learns
all tags at once, from all buildings. There are no perfect "twin" buildings, but every pair that differs in, say,
noise adds a little evidence to the noise tag. (Long version with examples: fig. 37.)

Tags **within the same district** (two buildings in the same district, everything else equal):

| Feature | Price tag |
|---|---|
| Built 2011–20 vs a 1961–90 prefab block | **+34 %** |
| Built before 1941 vs prefab | +4 % |
| Each km further walk to the metro | −3.5 % |
| Twice as many cafés and restaurants within a 7-minute walk | +2.7 % |
| 10 dB more tram and train noise at the façade | −1.8 % |
| 10 points more tree cover within 100 m (2025 LiDAR) | +1.1 % |
| Twice as many jobs reachable by transit | +0.3 % (could be zero) |
| Trams vs buses nearby | no difference |

Across the whole city, a building on the **right bank of the Vistula** sells **13.5 % below** an otherwise identical
one on the left bank.

**Building age matters most, location next, everything else is small.**

## Step 3: reputation

A second model (machine learning: gradient boosting) predicts each building's price **from its features alone,
without being told where it is**. Then: actual price minus predicted price = how much dearer or cheaper the building
is than its features suggest.

One building's gap is mostly noise (flat condition, a lucky sale). But when **almost every building in an area**
leans the same way, it's no longer noise. It's what the place is worth to buyers:

| Area | vs prediction |
|---|---|
| Praga-Północ | **−13 %** |
| Wola | −6 % |
| Żoliborz | +4 to +5 % |

**Since 2026-10-25 the model knows which side of the river a building is on.** Before that, Żoliborz looked +15 %,
but much of that was simply being on the left bank: the model had been comparing it with Praga, which is the same
distance from the centre. Praga-Północ keeps most of its discount even with the river side known.

## How good is it?

- It gets **60 %** of the price differences between buildings right, on buildings it has **never seen**, with their
  neighbours within 2 km hidden too. A typical miss is about 7–8 %.
- Knowing only the district gets 32 %.
- The other 40 % is mostly **inside the flat**: condition, finish, view, layout. No public data covers that.

**Is the model choice important?** We tried 9 kinds of model (`ml_compare.py`). Plain linear regression with the
district does as well as fancy machine learning (61 % vs 56–60 %). A neural network did worst (30 %). When the simple
model is this close, its price tags can be trusted. A flexible check ("double ML") keeps the tags' direction and
rough size. Age shrinks a bit (+28 %), and the metro tag becomes uncertain (−1 %, could be zero), so read the metro
number with care.

---

## Your questions

### Is it OK that distance to the Palace of Culture is so important?

**Yes. Location dominating is normal for housing models everywhere** ("location, location, location"). But the big
number overstates the Palace itself:

- It's the model's only general sense of **where** a building is, so it soaks up everything that differs across the
  city: the old core, prestige, which side of the river.
- Once the **district** is known, it adds less than one point of accuracy.
- Its importance fell from **42 % to 24 %** once the river side was added, because part of what it carried was
  really "left or right bank".
- It isn't a straight line: a steep premium within 2 km of the centre, then an almost flat plateau from 4 to 10 km,
  then a drop at the city's edge.

**Should we break it down?** We did, partly: the river side and job access by transit are now separate. What's left
in "distance to the centre" is mostly the core premium, a fair thing for one number to carry.

### Should districts be in the model? They're just administrative lines.

They're used in two different ways:

- **For price tags:** the district is a "compare like with like" control, so a Wola building is compared with Wola
  buildings. The lines are arbitrary, but the tags hardly change if we use the city's 143 smaller neighbourhoods
  instead (checked in `tram_test.py`, `transit_modes.py`, `frequency_story.py`).
- **For reputation:** the district is deliberately **left out**, so the model can't explain prices away by name.
  Districts only come back afterwards, to average the leftovers.

So the boundaries don't drive the results. They're a convenient grouping.

### Is there anything else worth adding?

Tested and **left out** (2026-10-25):

| Idea | Why not |
|---|---|
| Trams vs buses | no difference in price |
| Gated estates | the source data proved unreliable on the map |
| Nuisances (Radiowo landfill, Czajka sewage plant, Siekierki plant) | too few buildings nearby; mixed up with neighbourhood character |
| Crowding / *patodeweloperka* (buildings squeezed together) | no discount for crowded new buildings: buyers complain but pay the same |
| Finer job access (per building, not 250 m cells) | the local part is already in the walking distances; would take hours for a tiny change |
| Escarpment, school results | not how Warsaw works (high school by exam, not address) |
| City regeneration areas | would explain reputation away instead of measuring it; better as its own chart |

**Added:** right bank of the Vistula, jobs reachable by transit (morning peak).

What the model can't see is the inside of the flat. That isn't public, so the model is close to what public data
allows.

---

## Where the numbers come from

- **Prices:** every flat sale in Warsaw's official price register (RCN, GUGiK), 2006–2026.
- **Building age:** GIS Expert's map, plus Maksim Pashnin's dates (heritage registers), developer sales, OSM, BDOT10k
  and hand dating. Unknown for under 1 % of priced buildings.
- **Location:** OSM, the city's walking network with real fences and gates (TransitFrequency), GTFS timetables, EEA
  noise maps, 2025 LiDAR trees.

Code: `price_index.py` (step 1), `price_drivers.py` (steps 2 and 3), `ml_compare.py` (model comparison).
