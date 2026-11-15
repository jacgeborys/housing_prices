"""
Polish text for the report figures (REPORT_LANG=pl, see style.py). Translations follow sense, not word order.

EXACT: English string -> Polish. PATTERNS: (regex, template or function) for strings built with numbers.
KEEP: strings that stay as they are (names). numbers(): Polish number format on every string drawn, so Polish text
here may keep English number format (1,000 / 2.3 / 9K), it comes out as 1 000 / 2,3 / 9 tys.
Keys are the English strings as the scripts draw them: when a figure's numbers change, its key no longer matches and
the English string shows up in output/summary_pl/_untranslated.txt.
"""
import re

NBSP = " "

SRC = "Źródło: GUGiK, Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k."


def plural(n, one, few, many):
    """Polish noun after a number: 1 transakcja, 2 transakcje, 5 transakcji."""
    n = int(str(n).replace(",", "").replace(NBSP, ""))
    if n == 1:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


EXACT = {
    # --- shared
    "data gap\n2017–20": "luka w danych\n2017–20",
    "PLN per m²": "zł za m²",

    # --- 31 price ridgelines (cover)
    "Warsaw, in price": "Warszawa w cenach",
    "What a typical flat costs per m², 2025. Each line is a slice of the city;\nit rises where flats cost more.":
        "Ile kosztuje metr typowego mieszkania, 2025. Każda linia to przekrój miasta;\n"
        "wznosi się tam, gdzie mieszkania są droższe.",
    "Hedonic value of a typical 2-room resale flat, 2025, per building (1-2 sale buildings included, obvious outliers removed),\n"
    "smoothed over ~350 m; lines fade to gray where few flats sell.\n"
    "Line spacing 250 m; peak height grows 230 m per 1,000 PLN/m². Source: GUGiK Rejestr Cen Nieruchomości (RCN) 2006–2026, OSM, BDOT10k.":
        "Wartość typowego 2-pokojowego mieszkania z drugiej ręki w 2025 r. w każdym budynku (model hedoniczny; z budynkami o 1–2 transakcjach,\n"
        "bez skrajnych wartości), wygładzona na ~350 m; linie szarzeją tam, gdzie mieszkań sprzedaje się mało.\n"
        "Odstęp linii 250 m; szczyt rośnie o 230 m na każde 1,000 zł/m². Źródło: GUGiK, Rejestr Cen Nieruchomości (RCN) 2006–2026, OSM, BDOT10k.",

    # --- 32 price points
    "Where flats and houses sold": "Gdzie sprzedawano mieszkania i domy",
    "Flats: every building with a priced flat sale, 2006–2026 (bigger = more sales).\n"
    "Houses: every sale of known size, 2017–2026. Color: PLN per m², 2025.":
        "Mieszkania: każdy budynek ze sprzedażą, 2006–2026 (większy punkt = więcej transakcji).\n"
        "Domy: każda sprzedaż o znanej powierzchni, 2017–2026. Kolor: zł za m², 2025.",
    "Flats: a typical 2-room resale flat in 2025 per building. Houses: price per m² of living space, plot included, in 2025 prices.\n"
    "Cooperative-right flats are often missing from the register. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Mieszkania: typowe 2-pokojowe mieszkanie z drugiej ręki w 2025 r. w danym budynku. Domy: cena za m² powierzchni użytkowej\n"
        "z działką, w cenach z 2025 r. W rejestrze często brakuje mieszkań spółdzielczych. " + SRC,

    # --- 40 where flats are big
    "Where Warsaw's flats are big": "Gdzie w Warszawie są duże mieszkania",
    "Size and number of rooms of every flat sold since 2021, new and resale":
        "Powierzchnia i liczba pokoi każdego mieszkania sprzedanego od 2021 r., nowego i z drugiej ręki",
    "129,891 flat sales 2021–2026; hexagons with ≥ 15 sales. Score = average of a hexagon's rank on median m² and on mean rooms (the register's\n"
    "'izby'). Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "129,891 sprzedanych mieszkań 2021–2026; heksagony z ≥ 15 transakcjami. Wynik = średnia z miejsca heksagonu w rankingu mediany m²\n"
        "i średniej liczby pokoi (izby z rejestru). " + SRC,
    "A typical flat sold in Praga-Północ is 43 m² with 2.4 rooms on average; in Wawer, 70 m² and 3.3. The\n"
    "darker the hexagon, the bigger the flats and the more rooms.":
        "Typowe mieszkanie sprzedane na Pradze-Północ ma 43 m² i średnio 2.4 pokoju, w Wawrze 70 m² i 3.3.\n"
        "Im ciemniejszy heksagon, tym większe mieszkania i więcej pokoi.",
    "smaller flats": "mniejsze",
    "bigger flats, more rooms": "większe, więcej pokoi",
    "Typical flat by district: median m² on top, average rooms in the bar":
        "Typowe mieszkanie w dzielnicach: mediana m² nad słupkiem, średnia liczba pokoi w słupku",

    # --- 38 house map
    "One house sale for every 31 flat sales": "Na 31 sprzedanych mieszkań przypada jeden dom",
    "Houses ring the city; flats fill the middle. Every house sale 2017–2026, and every building with a flat sale":
        "Domy na obrzeżach, mieszkania w środku: sprzedaże domów 2017–2026 i budynki ze sprzedażą mieszkań",
    "Houses: 7,605 sales (recorded from 2017). Flats: 265,364 sales 2006–2026, gray = buildings with one. Ratio: 2021–25, when both are\n"
    "complete. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Domy: 7,605 transakcji (rejestrowane od 2017). Mieszkania: 265,364 transakcji 2006–2026, szare = budynki z co najmniej jedną.\n"
        "Proporcja z lat 2021–25, gdy dane obu rynków są pełne. " + SRC,
    "building with a flat sale (21,337)": "budynek ze sprzedanym mieszkaniem (21,337)",
    "one house sale (7,605)": "jedna sprzedaż domu (7,605)",
    "Wawer alone has 2,313 house sales, Białołęka 847, Wilanów 695: together 51% of all. Śródmieście\nhas 13.":
        "Sam Wawer to 2,313 sprzedanych domów, Białołęka 847, Wilanów 695: razem 51% wszystkich.\nŚródmieście: 13.",
    "Flat sales per year (thousands)": "Sprzedaż mieszkań rocznie (tys.)",
    "not recorded\nbefore 2017": "nierejestrowane\nprzed 2017",
    "House sales per year": "Sprzedaż domów rocznie",

    # --- 39 house cells
    "A house m² costs 9K in Wesoła and 24K in Żoliborz": "Metr domu: 9K w Wesołej, 24K na Żoliborzu",
    "Median price per m² of living space (plot included) of houses sold 2017–2026, in 2025 prices, per 300 m cell":
        "Mediana ceny za m² domu (z działką), sprzedaże 2017–2026 w cenach z 2025 r., oczka 300 m",
    "6,779 house sales (RCN, from 2017) in 2025 prices (house price index net of district). Cells with ≥ 3 sales colored (829 cells, 84% of\n"
    "sales). Area: deed or estimated from outline and floors. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "6,779 sprzedanych domów (RCN, od 2017) w cenach z 2025 r. (indeks cen domów po odjęciu dzielnicy). Kolorem oczka z ≥ 3 transakcjami\n"
        "(829, 84% transakcji). Powierzchnia: z aktu lub z obrysu i kondygnacji. " + SRC,
    "Median by district": "Mediana w dzielnicach",

    # --- 26 new flats on offer
    "Is buying from a developer a rip-off? Not really": "Czy kupując od dewelopera, przepłacasz? Raczej nie",
    "You pay for new, not for the developer: the typical project asks -2% vs a 5–15-year-old flat nearby":
        "Płacisz za nowość, nie za dewelopera: typowa inwestycja chce o 2% mniej niż 5–15-letnie mieszkanie obok",
    "Developers' asking prices (dane.gov.pl): 8,597 flats, 336 investments. Premium: median asking per m² vs the typical-flat value of priced\n"
    "2011–2020 buildings within 1 km. Resales: RCN 2024–26. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Ceny ofertowe deweloperów (dane.gov.pl): 8,597 mieszkań, 336 inwestycji. Premia: mediana ceny ofertowej za m² wobec typowego\n"
        "mieszkania z budynków 2011–2020 do 1 km. Drugi rynek: RCN 2024–26. " + SRC,
    "Asking, PLN/m²": "Cena ofertowa, zł/m²",
    "Where they are (circle size = flats on offer)": "Gdzie są (wielkość koła = mieszkania w ofercie)",
    "Asking vs a nearly new flat nearby": "Oferta wobec prawie nowego mieszkania obok",
    "flats on offer": "mieszkania w ofercie",
    "asking (new)": "cena ofertowa (nowe)",
    "typical resale": "typowe z drugiej ręki",
    "By district: supply (bars) and price per m² (dots)": "Dzielnice: podaż (słupki) i cena za m² (kropki)",
    "flat size, m²": "powierzchnia mieszkania, m²",
    "sold (resales 2024–26)": "sprzedane (drugi rynek 2024–26)",
    "on offer now (new)": "w ofercie (nowe)",
    "What's being built": "Co się buduje",
    "8,597 new flats on offer, median 16,555 PLN/m² and 50 m². The median investment asks 2% less per m²\n"
    "than a 5–15-year-old flat nearby; towers around Rondo Daszyńskiego ask 30–50k and lift Wola's\nmedian.":
        "8,597 nowych mieszkań w ofercie, mediana 16,555 zł/m² i 50 m². Typowa inwestycja chce za m² o 2% mniej\n"
        "niż 5–15-letnie mieszkanie obok; wieże przy Rondzie Daszyńskiego wołają 30–50 tys. i podbijają\nmedianę Woli.",

    # --- 16 flippers
    "Does flipping pay? +24% in a year, before costs": "Czy flipy się opłacają? +24% w rok, przed kosztami",
    "Flats resold within a year of purchase: where, how often, and for how much":
        "Mieszkania odsprzedane w ciągu roku od zakupu: gdzie, jak często i z jakim zyskiem",
    "Flip = the same register flat resold 14–365 days after its previous sale. Markups: flats bought second-hand only; 'beyond the market' nets\n"
    "out the city index. Map: 2021–25, ≥ 40 resales. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Flip = ten sam lokal z rejestru sprzedany ponownie 14–365 dni po poprzedniej transakcji. Narzut: tylko mieszkania kupione z drugiej ręki;\n"
        "„ponad rynek” = po odjęciu indeksu cen miasta. Mapa: 2021–25, ≥ 40 odsprzedaży. " + SRC,
    "2021–25: 4,151 flats were resold within a year of purchase (6 in 100 resales). Those bought second-\n"
    "hand sold for a median +28%, or +24% beyond the market, before renovation, tax and fees.":
        "2021–25: 4,151 mieszkań odsprzedano w ciągu roku od zakupu (6 na 100 transakcji z drugiej ręki). Kupione\n"
        "z drugiej ręki poszły w medianie o 28% drożej, 24% ponad rynek, przed kosztem remontu, podatkiem i opłatami.",
    "Flips per 100 resales": "Flipy na 100 odsprzedaży",
    "Top neighborhoods": "Najwięcej flipów",
    "Flips per 100 resales, by year": "Flipy na 100 odsprzedaży, rocznie",
    "median +28%": "mediana +28%",
    "Markup, as paid (bought second-hand, 2021–25)": "Narzut (zakup z drugiej ręki, 2021–25)",

    # --- 10 district anatomy
    "Why is Żoliborz dear? Location, then the name": "Dlaczego Żoliborz jest drogi? Lokalizacja, a potem marka",
    "How much of each district's price goes with location, transit and the building,\nand how much is just the name":
        "Ile z ceny w każdej dzielnicy wiąże się z lokalizacją, komunikacją i budynkiem,\na ile to sama nazwa",
    "Regression on all features, no district variable (14.0K buildings); bars = coefficient × distance from the city average. Right-bank flats\n"
    "sell ~13% below left-bank ones: not counted (mostly reputation). Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Regresja na wszystkich cechach bez zmiennej dzielnicy (14.0K budynków); słupki = współczynnik × odległość od średniej miasta. Prawy brzeg\n"
        "sprzedaje się ~13% taniej niż lewy: nieuwzględnione (to głównie renoma). " + SRC,
    "Contribution to the price gap vs the city average (≈ %)": "Udział w różnicy ceny wobec średniej miasta (≈ %)",
    "Total vs city average": "Razem wobec średniej miasta",
    "Distance to the center & jobs": "Odległość do centrum i miejsc pracy",
    "Metro & rail lines": "Metro i kolej",
    "Buses & trams": "Autobusy i tramwaje",
    "Cafés, shops & schools": "Kawiarnie, sklepy i szkoły",
    "Green, river & quiet": "Zieleń, rzeka i cisza",
    "Building age": "Wiek budynku",
    "Building size & type": "Wielkość i typ budynku",
    "Reputation (unexplained)": "Renoma (niewyjaśnione)",
    "Read one row: flats in Żoliborz cost +17% against the city average. Being close to the center\n"
    "and jobs goes with +8 points of that; the hatched gray, +11 points, is what no feature\nexplains: the place's reputation.":
        "Jak czytać wiersz: mieszkania na Żoliborzu są o 17% droższe od średniej miasta. Bliskość centrum\n"
        "i miejsc pracy daje z tego +8 punktów; zakreskowana szarość, +11 punktów, to coś, czego żadna\ncecha nie wyjaśnia: renoma miejsca.",

    # --- 11 transit x price
    "Where good transit is cheap": "Gdzie dobra komunikacja jest tania",
    "Bus, tram, metro and rail departures within a 5-minute walk, crossed with what a\ntypical flat costs":
        "Odjazdy autobusów, tramwajów, metra i kolei w zasięgu 5 minut pieszo,\nzestawione z ceną typowego mieszkania",
    "600 m hexagons with ≥ 5 priced buildings; median departures/day (TransitFrequency, 06–22 h, within a 5-minute walk) and median typical-\n"
    "flat price (2025 PLN/m²), each cut into thirds of the hexagons. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Heksagony 600 m z ≥ 5 wycenionymi budynkami; mediana odjazdów na dobę (TransitFrequency, 6–22, do 5 minut pieszo) i mediana ceny\n"
        "typowego mieszkania (2025, zł/m²), każda podzielona na tercyle heksagonów. " + SRC,
    "departures/day →": "odjazdy na dobę →",
    "price →": "cena →",
    "Well served, cheap  (66 hexagons)": "Dobra komunikacja, tanio  (66 heksagonów)",
    "Well served, dear  (79 hexagons)": "Dobra komunikacja, drogo  (79 heksagonów)",
    "Poorly served, dear  (76 hexagons)": "Słaba komunikacja, drogo  (76 heksagonów)",
    "Poorly served, cheap  (90 hexagons)": "Słaba komunikacja, tanio  (90 heksagonów)",

    # --- 12 price compass
    "Price compass from the Palace of Culture": "Kompas cen od Pałacu Kultury",
    "Typical-flat price per m², 2025, by direction and distance": "Cena m² typowego mieszkania w 2025 r. według kierunku i odległości",
    "Tile = sales-weighted median of typical-flat values (2025 PLN/m²), 1 km rings; tiles with under 15 priced buildings borrow the direction\n"
    "on each side, blank where none in their own. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Pole = mediana wartości typowego mieszkania ważona liczbą transakcji (2025, zł/m²), pierścienie co 1 km; pola z mniej niż 15 budynkami\n"
        "biorą też sąsiednie kierunki, puste, gdy we własnym nie ma żadnego. " + SRC,
    "PLN/m², typical flat 2025 (each band ≈ one ninth of the tiles)": "tys. zł/m², typowe mieszkanie 2025 (każdy przedział ≈ 1/9 pól)",
    "N": "Pn", "NE": "Pn-Wsch", "E": "Wsch", "SE": "Pd-Wsch", "S": "Pd", "SW": "Pd-Zach", "W": "Zach", "NW": "Pn-Zach",
    "Same distance, different direction: 4–7 km from the center, a typical flat to the southeast\n"
    "costs 17,427 PLN/m², to the east 14,013 (-20%).":
        "Ta sama odległość, inny kierunek: 4–7 km od centrum typowe mieszkanie na południowym wschodzie\n"
        "kosztuje 17,427 zł/m², na wschodzie 14,013 (-20%).",

    # --- 44 reputation glow (batch-2 cover)
    "Warsaw's reputation: what a model can't predict": "Renoma Warszawy: czego model nie przewidzi",
    "A model predicts what flats should cost from location, transit, cafés, parks, noise and the\n"
    "building. Warm: flats sell for more than the model predicts, cold: for less.":
        "Model przewiduje cenę mieszkań z lokalizacji, komunikacji, kawiarni, parków, hałasu\n"
        "i budynku. Ciepłe barwy: drożej, niż przewiduje model, zimne: taniej.",
    "−10%\nworse reputation": "−10%\ngorsza renoma",
    "+10%\nbetter reputation": "+10%\nlepsza renoma",
    "Each dot is a building with flat sales (14,018), bigger = more sales, colored by its area's gap within ~250 m.\n"
    "The model never saw the area it predicts. Source: GUGiK RCN, OSM, BDOT10k.":
        "Kropka = budynek z transakcjami (14,018); większa = więcej transakcji; kolor = różnica w okolicy ~250 m.\n"
        "Model nie widział obszaru, który przewiduje. Źródło: GUGiK RCN, OSM, BDOT10k.",

    # --- 18 building age
    "Does new always win? Over the prefabs, yes": "Czy nowe zawsze wygrywa? Z wielką płytą tak",
    "New buildings sell above their district, the 1960s–80s prefabs below; pre-war holds its own":
        "Nowe budynki są droższe od mediany dzielnicy, wielka płyta tańsza; przedwojenne trzymają poziom",
    "Construction decade: GIS Expert (mapy.gis-expert.pl) and others. Price: typical-flat value 2025 (≥ 3 sales) vs the district's median\n"
    "building. Map: 2011–20 vs 1961–90 buildings per 1.5 km hexagon. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Dekada budowy: GIS Expert (mapy.gis-expert.pl) i inne. Cena: wartość typowego mieszkania 2025 (≥ 3 transakcje) wobec mediany\n"
        "budynków dzielnicy. Mapa: budynki 2011–20 wobec 1961–90 w heksagonach 1.5 km. " + SRC,
    "before\n1901": "przed\n1901",
    "Typical flat vs the median building of its district, by construction decade":
        "Typowe mieszkanie wobec mediany dzielnicy, według dekady budowy",
    "pre-war": "przedwojenne",
    "post-war rebuild": "powojenna odbudowa",
    "prefab era": "wielka płyta",
    "after 1990": "po 1990",
    "New vs prefab": "Nowe a wielka płyta",
    "Premium of a 2011–20 building over a 1961–90 one nearby": "Premia budynku z lat 2011–20 nad pobliskim z 1961–90",
    "By district": "Dzielnice",
    "Same district, same size and floor: a flat in a 2011–20 building sells for about 33% more per m²\n"
    "than one in a 1961–90 prefab. Next door the gap runs from +95% in Śródmieście to +14% in Wesoła.":
        "Ta sama dzielnica, metraż i piętro: mieszkanie w budynku z lat 2011–20 jest za m² o ok. 33% droższe\n"
        "niż w bloku z lat 1961–90. Po sąsiedzku różnica sięga od +95% w Śródmieściu do +14% w Wesołej.",

    # --- 24 transit access
    "Where is the center, and who can reach it?": "Gdzie jest centrum i kto do niego dojedzie?",
    "Jobs reachable by public transport, door to door: walking, waiting, riding, changing":
        "Miejsca pracy w zasięgu komunikacji miejskiej, od drzwi do drzwi: dojście, czekanie, jazda, przesiadki",
    "Jobs = floor area (BDOT10k) by building category, 250 m cells. Door-to-door routing on the Warsaw timetable (7.10.2026, median of\n"
    "departures 07–09). Minutes: median over the district's priced buildings. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Miejsca pracy = powierzchnia (BDOT10k) według rodzaju budynku, oczka 250 m. Trasy od drzwi do drzwi wg rozkładu ZTM (7.10.2026, mediana\n"
        "odjazdów 7–9). Minuty: mediana dla wycenionych budynków dzielnicy. " + SRC,
    "ONZ /\nCentral Station\n37k": "ONZ /\nDw. Centralny\n37 tys.",
    "Rondo\nDaszyńskiego\n68k": "Rondo\nDaszyńskiego\n68 tys.",
    "Służewiec 22k": "Służewiec 22 tys.",
    "Jobs per 250 m cell": "Miejsca pracy na oczko 250 m",
    "Where the jobs are (circles: office cores)": "Gdzie jest praca (koła: skupiska biur)",
    "Jobs within 30 min": "Miejsca pracy w 30 min",
    "Jobs within 30 minutes, morning peak": "Miejsca pracy w 30 minut, poranny szczyt",
    "Minutes to the center (Palace of Culture) by public transport, 7–9 am, typical flat in the district":
        "Minuty do centrum (PKiN) komunikacją miejską, 7–9 rano, typowe mieszkanie w dzielnicy",
    "By public transport the center is 23 minutes away in Śródmieście and 61 in Wesoła.":
        "Komunikacją miejską do centrum jest 23 minuty ze Śródmieścia i 61 z Wesołej.",

    # --- 08 cafés
    "Do cafés raise prices? They follow them": "Czy kawiarnie podnoszą ceny? Raczej idą za nimi",
    "Flats cost more only where 40 or more venues are within a short walk. Cafés\nfollow wealth as much as they attract it":
        "Mieszkania są droższe dopiero tam, gdzie w krótkim spacerze jest 40 lub więcej lokali.\n"
        "Kawiarnie idą za zamożnością w tym samym stopniu, co ją przyciągają",
    "3,341 venues from OpenStreetMap (Oct 2026), per 600 m hexagon. Bars: typical-flat price net of district and other features, by venues\n"
    "within a 7-minute walk (+2.7% per doubling). An association, not a cause. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "3,341 lokali z OpenStreetMap (X 2026), w heksagonach 600 m. Słupki: cena typowego mieszkania po odjęciu dzielnicy i innych cech,\n"
        "wg lokali do 7 minut pieszo (+2.7% przy podwojeniu). To związek, nie przyczyna. " + SRC,
    "Venues per hexagon": "Lokale w heksagonie",
    "Price by venues nearby, all else equal": "Cena a lokale w pobliżu, przy pozostałych cechach bez zmian",
    "Cafés, restaurants and bars within a 7-minute walk": "Kawiarnie, restauracje i bary w zasięgu 7 minut pieszo",
    "none": "brak",

    # --- 09 noise
    "Does noise cost money? A little": "Czy hałas kosztuje? Trochę",
    "About -2% per 10 dB of tram and train noise, -0.9% for road noise":
        "Ok. -2% na każde 10 dB hałasu tramwajów i pociągów, -0.9% przy hałasie drogowym",
    "Map: EEA noise data (2026), the louder of road and rail (rail includes trams). Bars: typical-flat price net of district and other\n"
    "features, by tram & train noise only. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Mapa: dane o hałasie EEA (2026), głośniejszy z drogowego i szynowego (z tramwajami). Słupki: cena typowego mieszkania po odjęciu\n"
        "dzielnicy i innych cech, tylko wg hałasu tramwajów i pociągów. " + SRC,
    "Price by tram & train noise at the façade, all else equal":
        "Cena a hałas tramwajów i pociągów przy elewacji, przy pozostałych cechach bez zmian",
    "Tram & train noise (Lden) at the building": "Hałas tramwajów i pociągów (Lden) przy budynku",
    "below 55 dB": "poniżej 55 dB",

    # --- 20 trees
    "Trees: lovely, but barely priced": "Drzewa: piękne, ale prawie nie w cenie",
    "About +1% per 10 points of tree cover nearby. A park within 500 m does more: +2.6% per 10 ha":
        "Ok. +1% na 10 punktów procentowych zadrzewienia w pobliżu. Park do 500 m daje więcej: +2.6% na 10 ha",
    "Tree cover 2018 (Copernicus). Bars: typical-flat price net of district and other features, by tree cover within 100 m. Source: GUGiK\n"
    "Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Zadrzewienie 2018 (Copernicus). Słupki: cena typowego mieszkania po odjęciu dzielnicy i innych cech, wg zadrzewienia w promieniu 100 m.\n"
        + SRC,
    "Tree cover": "Zadrzewienie",
    "Price by tree cover within 100 m, all else equal": "Cena a zadrzewienie w promieniu 100 m, przy pozostałych cechach bez zmian",
    "Mean tree cover within 100 m of the building": "Średnie zadrzewienie w promieniu 100 m od budynku",

    # --- 06 cheaper than expected
    "What's an address worth? Up to 12% by district": "Ile jest wart adres? Do 12% w skali dzielnicy",
    "Red: flats sell above the model's prediction. Blue: below":
        "Czerwone: mieszkania sprzedają się drożej, niż przewiduje model. Niebieskie: taniej",
    "Same distance to the center, transit, cafés, green and noise, yet flats in Żoliborz sell ~12% above\n"
    "prediction and in Praga-Północ ~11% below. That gap is reputation.":
        "Ta sama odległość od centrum, komunikacja, kawiarnie, zieleń i hałas, a mieszkania na Żoliborzu\n"
        "są ~12% droższe od prognozy, na Pradze-Północ ~11% tańsze. Ta różnica to renoma.",
    "Median per 600 m hexagon (≥ 5 buildings): actual vs out-of-sample predicted value of a typical flat (spatial CV, no district variable). It\n"
    "holds reputation, condition, finish and views. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Mediana w heksagonie 600 m (≥ 5 budynków): faktyczna wartość typowego mieszkania wobec prognozy spoza próby (walidacja przestrzenna,\n"
        "bez zmiennej dzielnicy). Mieści renomę, stan, wykończenie i widoki. " + SRC,
    "as\nexpected": "zgodnie\nz prognozą",
    "as\npredicted": "zgodnie\nz prognozą",
    "vs prediction": "wobec prognozy",
    "as predicted": "zgodnie z prognozą",
    "Median building vs prediction, by district": "Mediana budynków wobec prognozy, wg dzielnic",

    # --- 15 estates
    "Which neighborhoods beat their prediction? Stary Żoliborz, by 23%": "Które osiedla biją prognozę? Stary Żoliborz, o 23%",
    "Neighborhoods that sell furthest above or below what their location, amenities and buildings predict":
        "Osiedla najdroższe i najtańsze względem tego, co przewidują ich położenie, otoczenie i budynki",
    "MSI neighborhoods with ≥ 10 priced buildings. Value = average gap between a building's typical-flat price and the out-of-sample\n"
    "prediction, shrunk toward 0 for small areas; bars = 95% interval. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Obszary MSI z ≥ 10 wycenionymi budynkami. Wartość = średnia różnica między ceną typowego mieszkania a prognozą spoza próby,\n"
        "przyciągnięta ku 0 dla małych obszarów; kreski = przedział 95%. " + SRC,
    "Price vs prediction (buildings in brackets)": "Cena wobec prognozy (w nawiasie liczba budynków)",

    # --- 14 reputation over time
    "Who gained since 2010? The inner city, whatever its reputation": "Kto zyskał od 2010 r.? Centrum, bez względu na renomę",
    "Same buildings, 2010–14 vs 2021–25: price change relative to Warsaw as a whole":
        "Te same budynki, 2010–14 wobec 2021–25: zmiana ceny względem całej Warszawy",
    "Resales only, buildings that existed before 2006; prices net of the city-wide index (0% = moved with the city). Map: 600 m hexagons.\n"
    "Scatter: neighborhoods with ≥ 8 buildings. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Tylko rynek wtórny, budynki istniejące przed 2006; ceny po odjęciu indeksu miasta (0% = tak jak miasto). Mapa: heksagony 600 m.\n"
        "Wykres punktowy: osiedla z ≥ 8 budynkami. " + SRC,
    "with\nthe city": "jak\nmiasto",
    "Reputation today: median price vs model prediction": "Renoma dziś: mediana ceny wobec prognozy modelu",
    "Change vs city, 2010–14 → 2021–25": "Zmiana wobec miasta, 2010–14 → 2021–25",
    "Neighborhoods: good and bad reputations gained alike; the two Pragas gain most":
        "Osiedla: dobra i zła renoma zyskały po równo; najwięcej obie Pragi",
    "Biggest gains against the city: Praga-Północ (+7%), Wola (+6%); biggest losses: Ursynów (-9%),\nUrsus (-8%).":
        "Najwięcej względem miasta zyskały Praga-Północ (+7%) i Wola (+6%), najwięcej straciły\nUrsynów (-9%) i Ursus (-8%).",

    # --- r98 other cities
    "Some other cities": "Inne miasta",
    "Same register, same method, same waves: Kraków, Gdańsk and Lublin.":
        "Ten sam rejestr, ta sama metoda, te same fale: Kraków, Gdańsk i Lublin.",
    "Typical 2-room resale flat, 2025, per building; each map on its own city's price scale (Warsaw: 12–24K+).\n"
    "Kraków's register is thin before 2021. Source: GUGiK RCN, BDOT10k.":
        "Typowe 2-pokojowe mieszkanie z drugiej ręki, 2025, w każdym budynku; każda mapa we własnej skali cen (Warszawa: 12–24K+).\n"
        "Krakowski rejestr jest ubogi przed 2021. Źródło: GUGiK RCN, BDOT10k.",

    # --- r01 what's in the data
    "What's in the data": "Co jest w danych",
    "Every flat and house sale in Warsaw's official price register, 2006–2026":
        "Każda sprzedaż mieszkania i domu w warszawskim Rejestrze Cen Nieruchomości, 2006–2026",
    "Deeds in full ownership. New = sold by the developer (price agreed a year or two before the deed). Flats 2017–20 nearly empty, houses\n"
    "recorded from 2017; cooperative flats are often missing; 2026 is partial. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Akty w pełnej własności. Nowe = sprzedane przez dewelopera (cena uzgodniona rok–dwa przed aktem). Mieszkania 2017–20 prawie puste,\n"
        "domy od 2017; mieszkań spółdzielczych często brak; 2026 niepełny. " + SRC,
    "265,364 flat sales: 157,473 resales and 107,544 from developers (347 without a market label), plus\n"
    "7,605 house sales (recorded from 2017). Most trade in Mokotów, fewest in Rembertów.":
        "265,364 sprzedanych mieszkań: 157,473 z drugiej ręki i 107,544 od deweloperów (347 bez oznaczenia\n"
        "rynku) oraz 7,605 domów (od 2017). Najwięcej transakcji na Mokotowie, najmniej w Rembertowie.",
    "thousand sales": "tys. transakcji",
    "flat, resale": "mieszkania: wtórny",
    "house, resale": "domy: wtórny",
    "flat, new (developer)": "mieszkania: pierwotny",
    "house, new (developer)": "domy: pierwotny",
    "Sales per year": "Sprzedaż rocznie",
    "thousand sales, 2006–2026": "tys. transakcji, 2006–2026",
    "Where they are": "W których dzielnicach",

    # --- r02 sizes
    "What sizes sell where? Small in Praga, big in Wawer": "Jakie metraże gdzie? Małe na Pradze, duże w Wawrze",
    "Flats sold by size, by district (2021–26) and across the years":
        "Sprzedane mieszkania według metrażu: w dzielnicach (2021–26) i w kolejnych latach",
    "RCN flat sales in full ownership, new and resale together. Size = usable floor area from the deed. Years with fewer than 300 sales left\n"
    "out (2017–20, Warsaw's register data gap). Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Sprzedaże mieszkań w pełnej własności z RCN, nowe i z drugiej ręki razem. Metraż = powierzchnia użytkowa z aktu. Lata z mniej niż 300\n"
        "transakcjami pominięte (2017–20, luka w danych rejestru). " + SRC,
    "Small flats (under 35 m²) are 27% of sales in Praga-Północ but 3% in Wesoła; flats of 70 m² or more\n"
    "are most common in Wawer and Wilanów. Micro flats under 25 m² are rare: 3% of all sales.":
        "Małe mieszkania (do 35 m²) to 27% transakcji na Pradze-Północ, ale 3% w Wesołej; te od 70 m²\n"
        "są najczęstsze w Wawrze i Wilanowie. Mikro poniżej 25 m² to rzadkość: 3% wszystkich transakcji.",
    "micro, under 25 m²": "mikro, do 25 m²",
    "small, 25–35": "małe, 25–35",
    "compact, 35–50": "kompaktowe, 35–50",
    "medium, 50–70": "średnie, 50–70",
    "spacious, 70–100": "przestronne, 70–100",
    "large, 100+": "duże, 100+",
    "micro": "mikro", "small": "małe", "compact": "kompaktowe", "medium": "średnie", "spacious": "przestronne",
    "large": "duże",
    "Share of flats sold, 2021–26": "Udział w sprzedanych mieszkaniach, 2021–26",
    "Share of flats sold, by year": "Udział w sprzedanych mieszkaniach, rocznie",

    # --- 01 prices over time
    "Record prices? In real terms, just past 2008": "Rekordowe ceny? Realnie ledwie powyżej 2008",
    "Flats cost 2.3× their 2013 price, but after inflation only 1.5×: just back at the 2008 peak":
        "Mieszkania kosztują 2.3× tyle co w 2013, ale po inflacji tylko 1.5×: tyle co na szczycie z 2008",
    "The same kind of flat each quarter (within-building comparisons). Real = deflated with GUS CPI to 2025 PLN. Dashed: 2017–20, too few sales\n"
    "in Warsaw's register data. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Ten sam rodzaj mieszkania co kwartał (porównania w obrębie budynków). Realnie = po deflacji CPI GUS, zł z 2025. Przerywana: 2017–20,\n"
        "za mało transakcji w rejestrze. " + SRC,
    "Price as paid (nominal)": "Cena transakcyjna (nominalna)",
    "In today's money (real, 2025 PLN)": "W dzisiejszych pieniądzach (realnie, zł z 2025)",
    "PLN per m², typical flat": "zł za m², typowe mieszkanie",
    "2008 peak, in today's money: 16,722": "Szczyt z 2008 w dzisiejszych pieniądzach: 16,722",
    "mid-2026": "poł. 2026",
    "price as paid\n2013 → 2025": "cena transakcyjna\n2013 → 2025",
    "after inflation\n(today's money), 2013 → 2025": "po inflacji\n(dzisiejsze pieniądze), 2013 → 2025",
    "2025 vs the 2008 peak,\nin today's money": "2025 wobec szczytu z 2008,\nw dzisiejszych pieniądzach",
    "Warsaw ×1.4": "Warszawa ×1.4",
    "And elsewhere? After inflation, 2016 → 2025": "A gdzie indziej? Po inflacji, 2016 → 2025",

    # --- 03 what drives price
    "What goes with the price of a Warsaw flat": "Co idzie w parze z ceną warszawskiego mieszkania",
    "Buildings compared within the same district: how much more or less per m²":
        "Budynki porównane w obrębie dzielnicy: o ile więcej lub mniej za m²",
    "14,018 buildings with ≥ 3 flat sales; typical flat (date, floor, size removed); lines = 95% range. Złoty: % × the median building's price,\n"
    "50 m². The features explain 62% of price differences. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "14,018 budynków z ≥ 3 sprzedażami mieszkań; typowe mieszkanie (bez efektu daty, piętra, metrażu); kreski = przedział 95%. Złote: % ×\n"
        "cena mediany budynków, 50 m². Cechy wyjaśniają 62% różnic cen. " + SRC,
    "+11%  (off scale)": "+11%  (poza skalą)",
    "+33%  (off scale)": "+33%  (poza skalą)",
    "≈ on a\n50 m² flat": "≈ przy\n50 m²",
    "Each km farther to walk to the\nmetro": "Każdy km dalej pieszo\ndo metra",
    "Each km farther from the\ncenter": "Każdy km dalej\nod centrum",
    "Tram & train noise 10 dB\nlouder": "Hałas tramwajów i pociągów\no 10 dB głośniejszy",
    "Twice as many Żabka shops\nwithin a 7-min walk": "Dwa razy więcej Żabek\nw 7 min pieszo",
    "Road noise 10 dB louder": "Hałas drogowy o 10 dB głośniejszy",
    "Building 10 m taller": "Budynek wyższy o 10 m",
    "Twice as many bus and tram\ndepartures": "Dwa razy więcej odjazdów\nautobusów i tramwajów",
    "Built 1941–1960 (vs a 1961–90\nprefab block)": "Zbudowany 1941–1960 (wobec\nbloku z lat 1961–90)",
    "Twice the building footprint": "Dwa razy większy obrys budynku",
    "10 points more tree cover\nwithin 100 m": "10 pkt proc. więcej drzew\ndo 100 m",
    "10 ha more park or forest\nwithin 500 m": "10 ha więcej parku lub lasu\ndo 500 m",
    "Twice as many cafés,\nrestaurants and bars within a\n7-min walk": "Dwa razy więcej kawiarni,\nrestauracji i barów\nw 7 min pieszo",
    "New building (sold by the\ndeveloper)": "Nowy budynek (sprzedaż\nod dewelopera)",
    "Built 2011–2020 (vs a 1961–90\nprefab block)": "Zbudowany 2011–2020 (wobec\nbloku z lat 1961–90)",
    "Change in price per m²": "Zmiana ceny za m²",
    "Take two flats of the same size on the same floor, in the same district. One is in a 2011–20\n"
    "building, the other in a 1961–90 prefab block: the newer one costs about 33% more per m²,\n"
    "roughly 259k zł on a 50 m² flat. Every bar below is a comparison like that.":
        "Weźmy dwa mieszkania tej samej wielkości, na tym samym piętrze, w tej samej dzielnicy. Jedno\n"
        "w budynku z lat 2011–20, drugie w bloku z lat 1961–90: nowsze kosztuje za m² ok. 33% więcej,\n"
        "czyli ok. 259 tys. zł przy 50 m². Każdy słupek poniżej to takie porównanie.",

    # --- 23 blind spot
    "What can't the register see? 1 in 6 flats": "Czego nie widzi rejestr? 1 na 6 mieszkań",
    "Cooperative-right flats (spółdzielcze własnościowe prawo) often don't make it\ninto the register, and rented city blocks rarely sell":
        "Spółdzielcze własnościowe prawa do lokalu często nie trafiają do rejestru,\na mieszkania komunalne rzadko są sprzedawane",
    "Multi-family buildings only (3+ flats, BDOT10k / OSM); flats estimated from footprint × stories. Few sales = under 25% of the city's usual\n"
    "sales per flat. The register mostly covers flats in full ownership. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Tylko budynki wielorodzinne (3+ mieszkań, BDOT10k / OSM); mieszkania szacowane z obrysu × kondygnacji. Mało transakcji = poniżej 25%\n"
        "średniej miasta na mieszkanie. Rejestr obejmuje głównie pełną własność. " + SRC,
    "1 in 6": "1 na 6",
    "flats sits in a building with\nno recorded sale at all": "mieszkań jest w budynku bez żadnej\nzarejestrowanej sprzedaży",
    "more are in buildings with\nonly a handful of sales": "kolejnych jest w budynkach\nz ledwie kilkoma sprzedażami",
    "Share of the district's flats": "Udział mieszkań dzielnicy",
    "No recorded sale": "Brak sprzedaży",
    "Few sales": "Mało transakcji",
    "Covered": "Pokryte",

    # --- 35 house or flat
    "A house costs 2.2× a flat, a square meter a third less": "Dom kosztuje 2.2× tyle co mieszkanie, metr o 1/3 mniej",
    "Houses and resold flats sold 2023–2026 in the 14 districts with at least 40 house sales":
        "Domy i mieszkania z drugiej ręki, 2023–2026, w 14 dzielnicach z min. 40 sprzedanymi domami",
    "Houses: 2,629 sales with plot (RCN, from 2017); flats: 33,596 resales, same districts and years. House area: the deed's (33%) or estimated\n"
    "from the building outline and floors. Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.":
        "Domy: 2,629 transakcji z działką (RCN, od 2017); mieszkania: 33,596 z drugiej ręki, te same dzielnice i lata. Powierzchnia domu: z aktu\n"
        "(33%) lub szacowana z obrysu i kondygnacji. " + SRC,
    "In total a median house costs 1.6 M and a flat 0.73 M. Per m² it flips: 9.9K for a house, plot\n"
    "included, 15.1K for a flat. A house buys much more space, and the land comes with it.":
        "W sumie typowy dom kosztuje 1.6 M, a mieszkanie 0.73 M. Za metr odwrotnie: 9.9K za dom z działką,\n"
        "15.1K za mieszkanie. Dom daje dużo więcej przestrzeni, a ziemia jest w pakiecie.",
    "price paid, PLN": "zapłacona cena, zł",
    "Resold flat": "Mieszkanie z drugiej ręki",
    "House (with plot)": "Dom (z działką)",
    "What you pay in total": "Ile płacisz w sumie",
    "price per m² of living space, PLN": "cena za m² powierzchni użytkowej, zł",
    "What a square meter costs": "Ile kosztuje metr",
}

PATTERNS = [
    (r" ([\d.]+)K  \(([\d,]+) sales\)",
     lambda m: f" {m[1]}K  ({m[2]} {plural(m[2], 'transakcja', 'transakcje', 'transakcji')})"),
    (r"(\d+\.\d)K", lambda m: m[1]),                  # compass colorbar ticks: 14.0K -> 14,0 (label says tys. zł/m²)
    (r"colors: (.+)", lambda m: "kolory: " + m[1]),
]

KEEP = {
    "© 2026 Jacek Gęborys",
    "Śródmieście", "Żoliborz", "Mokotów", "Wilanów", "Praga-Północ", "Praga-Południe", "Białołęka", "Ursynów", "Bemowo",
    "Wawer", "Ursus", "Targówek", "Wola", "Ochota", "Bielany", "Włochy", "Wesoła", "Rembertów",
}


def numbers(s):
    s = re.sub(r"(\d),(?=\d{3}\b)", r"\1" + NBSP, s)                  # 17,405 -> 17 405
    s = re.sub(r"(\d),(?=\d{3}\b)", r"\1" + NBSP, s)                  # 1,234,567 (second pass)
    s = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?!\d|\.\d)", r"\1,\2", s)     # 2.3 -> 2,3 (not dates like 1.11.2026)
    s = re.sub(r"\b(\d+(?:,\d+)?)\s?[Kk]\b", r"\1" + NBSP + "tys.", s)   # 600K, 15k -> 600 tys. (not BDOT10k)
    s = re.sub(r"\b(\d+(?:,\d+)?)\s?M\b", r"\1" + NBSP + "mln", s)       # 1.6 M -> 1,6 mln
    s = re.sub(r"\bPLN/m²", "zł/m²", s)
    s = re.sub(r"(\d)\s?PLN\b", r"\1" + NBSP + "zł", s)
    return s
