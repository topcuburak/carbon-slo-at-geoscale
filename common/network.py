"""Network topology for the distributed AI inference simulation.

Defines scheduler groups (geographic clusters of datacenters) and
inter-group RTT values based on real-world approximate latencies.

8 scheduler groups, variable DCs per group (104 total, backed by
real carbon-intensity data from ElectricityMap).

Groups:
  - europe_north:  Nordic + British Isles + Benelux + Baltics (15 DCs)
  - europe_south:  Continental Western/Central/Southern Europe   (18 DCs)
  - us_east:       US East Coast + Great Lakes ISOs              (8 DCs)
  - us_south:      US Southeast + Gulf + Texas ISOs              (8 DCs)
  - us_west:       US Pacific + Mountain West ISOs               (15 DCs)
  - middle_east:   Eastern Mediterranean + Balkans + S. Africa   (7 DCs)
  - asia:          East/SE Asia + Oceania + Indian subcontinent  (26 DCs)
  - south_america: Latin America                                 (7 DCs)

RTT sources (one-way, approximate):
  - Intra-group (~2-10ms): same-region DC interconnect
  - Europe North <-> Europe South (~15ms)
  - US East <-> US South (~25ms), US East <-> US West (~65ms)
  - Europe <-> US East (~80-85ms), Europe <-> US West (~135-140ms)
  - Europe <-> Middle East (~55-65ms)
  - US <-> Middle East (~130-190ms)
  - US West <-> Asia (~120ms), Europe <-> Asia (~160-175ms)
  - Middle East <-> Asia (~95ms)
  - South America <-> US East (~120ms), <-> US South (~110ms)
  - South America <-> Europe (~185-200ms), <-> Asia (~280ms)
  Ref: https://www.cloudping.co/grid, AWS inter-region latency dashboard
"""

import json
import os
import random
from typing import Dict, List, Tuple

# ── Full pool of available DCs per group ────────────────────────────
# Each DC name maps to a carbon-data directory via DC_TO_CARBON_PATH.
# Not all DCs are used in every run — sweep scripts select a subset.

SCHEDULER_GROUPS: Dict[str, list] = {
    # ── Europe North: Nordic + British Isles + Benelux + Baltics ──
    "europe_north": [
        "DK_East", "DK_West", "Estonia", "Finland",
        "GreatBritain", "Ireland", "Latvia", "Lithuania",
        "Netherlands", "Norway", "SE_North", "SE_Central",
        "SE_South", "Belgium", "Poland",
    ],
    # ── Europe South: Continental Western/Central/Southern ──
    "europe_south": [
        "Austria", "Czechia", "France", "Germany",
        "Hungary", "IT_North", "IT_CentralNorth", "IT_CentralSouth",
        "IT_South", "IT_Sardinia", "IT_Sicily", "Portugal",
        "Serbia", "Slovakia", "Slovenia", "Spain",
        "Switzerland", "Croatia",
    ],
    # ── US East: East Coast + Great Lakes ISOs ──
    "us_east": [
        "US_ISONE", "US_NYISO", "US_PJM", "US_MISO",
        "US_DukeCarolinas", "US_DukeProgressE", "US_DukeProgressW",
        "US_SCEG",
    ],
    # ── US South: Southeast + Gulf + Texas ISOs ──
    "us_south": [
        "US_Southern", "US_Tallahassee", "US_FLMunicipal", "US_FPL",
        "US_Gainesville", "US_TVA", "US_ERCOT", "US_SPP",
    ],
    # ── US West: Pacific + Mountain West ISOs ──
    "us_west": [
        "US_APS", "US_Avista", "US_BANC", "US_BPA",
        "US_CAISO", "US_ElPaso", "US_IdahoPower", "US_LADWP",
        "US_NWEnergy", "US_PacifiCorpE", "US_PGE", "US_PSCo",
        "US_Turlock", "US_WAPA", "US_AssocElec",
    ],
    # ── Middle East: Eastern Mediterranean + Balkans + S. Africa ──
    "middle_east": [
        "Israel", "Turkey", "Cyprus", "Greece",
        "Bulgaria", "Romania", "SouthAfrica",
    ],
    # ── Asia: East/SE Asia + Oceania + Indian subcontinent ──
    "asia": [
        "AU_NSW", "AU_NT", "AU_QLD", "AU_VIC",
        "IN_East", "IN_NorthEast", "IN_North",
        "IN_South", "IN_West", "JP_Chubu",
        "JP_Chugoku", "JP_Hokkaido", "JP_Hokuriku", "JP_Kansai",
        "JP_Okinawa", "JP_Tohoku", "JP_Tokyo", "PH_Luzon",
        "PH_Mindanao", "PH_Visayas", "Singapore", "SouthKorea",
        "Taiwan", "NewZealand",
    ],
    # ── South America: Latin America ──
    "south_america": [
        "BR_Central", "BR_North", "BR_NorthEast", "BR_South",
        "Chile", "Peru", "Nicaragua",
    ],
}

# ── DC name → carbon data directory path (relative to carbon_data/) ─
DC_TO_CARBON_PATH: Dict[str, str] = {
    # Europe North
    "DK_East":       "denmark/east_denmark",
    "DK_West":       "denmark/west_denmark",
    "Estonia":       "estonia",
    "Finland":       "finland",
    "GreatBritain":  "great_britain",
    "Ireland":       "ireland",
    "Latvia":        "latvia",
    "Lithuania":     "lithuania",
    "Netherlands":   "netherlands",
    "Norway":        "norway/norway",
    "SE_North":      "sweden/north_central_sweden",
    "SE_Central":    "sweden/south_central_sweden",
    "SE_South":      "sweden/south_sweden",
    "Belgium":       "belgium",
    "Poland":        "poland",
    # Europe South
    "Austria":       "austria",
    "Czechia":       "czechia",
    "France":        "france",
    "Germany":       "germany",
    "Hungary":       "hungary",
    "IT_North":      "italy/north_italy",
    "IT_CentralNorth": "italy/central_north_italy",
    "IT_CentralSouth": "italy/central_south_italy",
    "IT_South":      "italy/south_italy",
    "IT_Sardinia":   "italy/sardinia",
    "IT_Sicily":     "italy/sicily",
    "Portugal":      "portugal",
    "Serbia":        "serbia",
    "Slovakia":      "slovakia",
    "Slovenia":      "slovenia",
    "Spain":         "spain",
    "Switzerland":   "switzerland",
    "Croatia":       "crotia",
    # US East
    "US_ISONE":      "USA/iso_new_england_inc",
    "US_NYISO":      "USA/new_york_independent_system_operator",
    "US_PJM":        "USA/pjm_interconnection_LLC",
    "US_MISO":       "USA/midcontinent_independent_transmission_system_operator",
    "US_DukeCarolinas":  "USA/duke_energy_carolinas",
    "US_DukeProgressE":  "USA/duke_energy_progress_east",
    "US_DukeProgressW":  "USA/duke_energy_progress_west",
    "US_SCEG":       "USA/south_carolina_electric_gas_company",
    # US South
    "US_Southern":   "USA/southern_company_services_inc_trans",
    "US_Tallahassee": "USA/city_of_tallahassee",
    "US_FLMunicipal": "USA/florida_municipal_power_pool",
    "US_FPL":        "USA/florida_power_and_light_company",
    "US_Gainesville": "USA/gainesville_regional_utilities",
    "US_TVA":        "USA/tennessee_valley_authority",
    "US_ERCOT":      "USA/electric_reliability_council_of_texas",
    "US_SPP":        "USA/southwest_power_pool",
    # US West
    "US_APS":        "USA/arizona_public_service_company",
    "US_Avista":     "USA/avista_corporation",
    "US_BANC":       "USA/balancing_authority_of_northern_california",
    "US_BPA":        "USA/bonneville_power_administration",
    "US_CAISO":      "USA/california_independent_system_operator",
    "US_ElPaso":     "USA/el_paso_electric_company",
    "US_IdahoPower": "USA/idaho_power_company",
    "US_LADWP":      "USA/los_angeles_dept_of_water_and_power",
    "US_NWEnergy":   "USA/northwestern_energy",
    "US_PacifiCorpE": "USA/pacificorp_east",
    "US_PGE":        "USA/portland_general_electric_company",
    "US_PSCo":       "USA/public_service_company_of_colorado",
    "US_Turlock":    "USA/turlock_irrigation_district",
    "US_WAPA":       "USA/western_area_power_administration_rocky_mountain_region",
    "US_AssocElec":  "USA/associated_electric_cooperative",
    # Middle East
    "Israel":        "israel",
    "Turkey":        "turkey",
    "Cyprus":        "cyprus",
    "Greece":        "greece",
    "Bulgaria":      "bulgaria",
    "Romania":       "romania",
    "SouthAfrica":   "south_africa",
    # Asia
    "AU_NSW":        "australia/new_south_wales",
    "AU_NT":         "australia/nothern_territory",
    "AU_QLD":        "australia/queensland",
    "AU_VIC":        "australia/victoria",
    "IN_East":       "india/eastern_india",
    "IN_NorthEast":  "india/north_eastern_india",
    "IN_North":      "india/northern_india",
    "IN_South":      "india/southern_india",
    "IN_West":       "india/western_india",
    "JP_Chubu":      "japan/chubu",
    "JP_Chugoku":    "japan/chugoku",
    "JP_Hokkaido":   "japan/hokkaido",
    "JP_Hokuriku":   "japan/hokuriku",
    "JP_Kansai":     "japan/kansai",
    "JP_Okinawa":    "japan/okinawa",
    "JP_Tohoku":     "japan/tohoku",
    "JP_Tokyo":      "japan/tokyo",
    "PH_Luzon":      "philippines/luzon",
    "PH_Mindanao":   "philippines/mindanao",
    "PH_Visayas":    "philippines/visayas",
    "Singapore":     "singapore",
    "SouthKorea":    "south_korea",
    "Taiwan":        "taiwan",
    "NewZealand":    "new_zeland",
    # South America
    "BR_Central":    "brazil/central_brazil",
    "BR_North":      "brazil/north_brazil",
    "BR_NorthEast":  "brazil/north_east_brazil",
    "BR_South":      "brazil/south_brazil",
    "Chile":         "chile",
    "Peru":          "peru",
    "Nicaragua":     "nicaragua",
}

# ── Reverse lookups ─────────────────────────────────────────────────

# Region name -> scheduler group name
REGION_TO_GROUP: Dict[str, str] = {}
for _group, _regions in SCHEDULER_GROUPS.items():
    for _region in _regions:
        REGION_TO_GROUP[_region] = _group

# Sub-group -> the geographic group it was split from. Empty unless a run's
# topology file splits a region between schedulers (see _apply_subgroups).
GROUP_PARENT: Dict[str, str] = {}

# ── Round-trip time in ms between scheduler groups (symmetric) ──────
# 8 groups → 8 self + 28 cross = 36 entries.
#
# These are ROUND TRIP, from cloudping.co's one-year p50 between AWS
# regions, which is what that site measures. They were labelled one-way
# and then doubled by every consumer, so the simulated network cost was
# twice reality: europe_north <-> us_east was charged 170 ms against a
# published 75-90, and a metro-local hop 4 ms against 1-2. Measured on
# carbon_aware:0.0 at 3200 req/s, 87% of requests were forwarded and paid
# 438 ms of it, 15% of the TTFT budget, worth 5.5 points of SLO.
#
# A forward therefore costs ONE of these values end to end: half outbound,
# half on the return.

_RTT_MS: Dict[Tuple[str, str], float] = {
    # ── self (intra-group) ──
    ("europe_north",  "europe_north"):  2.0,
    ("europe_south",  "europe_south"):  2.0,
    ("us_east",       "us_east"):       2.0,
    ("us_south",      "us_south"):      2.0,
    ("us_west",       "us_west"):       2.0,
    ("middle_east",   "middle_east"):   2.0,
    ("asia",          "asia"):          2.0,
    ("south_america", "south_america"): 10.0,
    # ── Europe internal ──
    ("europe_north", "europe_south"):   15.0,
    # ── US internal ──
    ("us_east",  "us_south"):           25.0,
    ("us_east",  "us_west"):            65.0,
    ("us_south", "us_west"):            45.0,
    # ── Europe <-> US ──
    ("europe_north", "us_east"):        85.0,
    ("europe_north", "us_south"):       95.0,
    ("europe_north", "us_west"):        140.0,
    ("europe_south", "us_east"):        80.0,
    ("europe_south", "us_south"):       90.0,
    ("europe_south", "us_west"):        135.0,
    # ── Europe <-> Middle East ──
    ("europe_north", "middle_east"):    65.0,
    ("europe_south", "middle_east"):    55.0,
    # ── US <-> Middle East ──
    ("us_east",  "middle_east"):        130.0,
    ("us_south", "middle_east"):        140.0,
    ("us_west",  "middle_east"):        190.0,
    # ── Europe <-> Asia ──
    ("europe_north", "asia"):           175.0,
    ("europe_south", "asia"):           160.0,
    # ── US <-> Asia ──
    ("us_east",  "asia"):              190.0,
    ("us_south", "asia"):              180.0,
    ("us_west",  "asia"):              120.0,
    # ── Middle East <-> Asia ──
    ("middle_east", "asia"):            95.0,
    # ── South America <-> US ──
    ("south_america", "us_east"):       120.0,
    ("south_america", "us_south"):      110.0,
    ("south_america", "us_west"):       160.0,
    # ── South America <-> Europe ──
    ("south_america", "europe_north"):  200.0,
    ("south_america", "europe_south"):  185.0,
    # ── South America <-> Middle East ──
    ("south_america", "middle_east"):   250.0,
    # ── South America <-> Asia ──
    ("south_america", "asia"):          280.0,
}

def get_rtt_ms(
    group_a: str,
    group_b: str,
    jitter_pct: float = 0.0,
) -> float:
    """Return the round-trip time in ms between two scheduler groups.

    Args:
        jitter_pct: Random jitter as a fraction (0.0-1.0). E.g. 0.2
            means the returned value is uniformly sampled from
            [base * (1 - 0.2), base * (1 + 0.2)].
    """
    # A sub-group sits where its region does: two halves of one region are the
    # region's own round trip apart, and each is as far from the rest as the whole.
    key = (GROUP_PARENT.get(group_a, group_a), GROUP_PARENT.get(group_b, group_b))
    if key not in _RTT_MS:
        key = (key[1], key[0])
    if key not in _RTT_MS:
        raise ValueError(f"No RTT defined for groups: {group_a} <-> {group_b}")

    base = _RTT_MS[key]
    if jitter_pct > 0.0 and base > 0.0:
        lo = base * (1.0 - jitter_pct)
        hi = base * (1.0 + jitter_pct)
        return random.uniform(lo, hi)
    return base


def rtt_pair_key(group_a: str, group_b: str) -> str:
    """Canonical sorted key for an RTT pair, e.g. 'asia|europe_north'."""
    a, b = sorted([group_a, group_b])
    return f"{a}|{b}"


def get_all_group_names() -> List[str]:
    """Return sorted list of all scheduler group names."""
    return sorted(SCHEDULER_GROUPS.keys())


def get_all_base_rtts() -> Dict[str, float]:
    """Return {canonical_key: base_rtt_ms} for all group pairs (including self).

    Round trip, not one way. See the table's own note.
    """
    result = {}
    groups = get_all_group_names()
    for i, g1 in enumerate(groups):
        for g2 in groups[i:]:
            key = rtt_pair_key(g1, g2)
            result[key] = get_rtt_ms(g1, g2)
    return result


def get_rtt_for_regions(
    region_a: str,
    region_b: str,
    jitter_pct: float = 0.0,
) -> float:
    """Return the round-trip time in ms between two datacenter regions."""
    group_a = REGION_TO_GROUP[region_a]
    group_b = REGION_TO_GROUP[region_b]
    return get_rtt_ms(group_a, group_b, jitter_pct=jitter_pct)


# ── Several schedulers in one region ─────────────────────────────────
# A run's topology file may split a region between schedulers so that the
# scheduler count changes while the DC set and the geography stay the same:
#
#   "topology":      {"europe_north_a": ["Netherlands", "Belgium"],
#                     "europe_north_b": ["DK_East", "DK_West"], ...},
#   "group_parents": {"europe_north_a": "europe_north",
#                     "europe_north_b": "europe_north", ...}
#
# Each sub-group owns exactly its listed DCs, and get_rtt_ms prices it as its
# parent region. The sweep exports the run's topology path as
# SIM_TOPOLOGY_FILE, so every process applies the same split at import; a file
# without "group_parents" changes nothing.

def _apply_subgroups(path: str) -> None:
    with open(path) as f:
        topo = json.load(f)
    parents = topo.get("group_parents") or {}
    if not parents:
        return
    groups = topo.get("topology", {})
    for sub, parent in parents.items():
        if parent not in SCHEDULER_GROUPS or parent in parents:
            raise ValueError(f"{path}: sub-group {sub!r} names unknown parent {parent!r}")
        if parent in groups:
            raise ValueError(f"{path}: {parent!r} is split into sub-groups and cannot "
                             f"also run as a scheduler group")
        dcs = groups.get(sub)
        if not dcs:
            raise ValueError(f"{path}: sub-group {sub!r} lists no DCs")
        stray = [dc for dc in dcs if REGION_TO_GROUP.get(dc) != parent]
        if stray:
            raise ValueError(f"{path}: {stray} are not in {parent!r}, so {sub!r} "
                             f"would be priced at the wrong RTT")
    for sub, parent in parents.items():
        SCHEDULER_GROUPS[sub] = list(groups[sub])
        GROUP_PARENT[sub] = parent
        for dc in groups[sub]:
            REGION_TO_GROUP[dc] = sub


if os.environ.get("SIM_TOPOLOGY_FILE"):
    _apply_subgroups(os.environ["SIM_TOPOLOGY_FILE"])
