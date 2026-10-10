"""Carbon intensity data loader.

Reads hourly CSV files from the carbon_data directory and provides
O(1) lookup by region name + UTC datetime.

Region names match common/network.py SCHEDULER_GROUPS (102 DCs).

Hong Kong and Indonesia were removed on 2026-09-02: their hourly files
are a constant rather than a measurement. See
carbon_intensity/_retired/README.md.
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Region name -> (subdirectory path, zone_id prefix for filename)
# Matches DC names in common/network.py SCHEDULER_GROUPS.
# Filename format: {zone_prefix}_{year}_hourly.csv
REGION_CSV_MAP: Dict[str, Tuple[str, str]] = {
    # ── Europe North (15) ────────────────────────────────────────────
    "DK_East":       ("denmark/east_denmark",          "DK-DK2"),
    "DK_West":       ("denmark/west_denmark",          "DK-DK1"),
    "Estonia":       ("estonia",                       "EE"),
    "Finland":       ("finland",                       "FI"),
    "GreatBritain":  ("great_britain",                 "GB"),
    "Ireland":       ("ireland",                       "IE"),
    "Latvia":        ("latvia",                        "LV"),
    "Lithuania":     ("lithuania",                     "LT"),
    "Netherlands":   ("netherlands",                   "NL"),
    "Norway":        ("norway/norway",                 "NO"),
    "SE_North":      ("sweden/north_central_sweden",   "SE-SE2"),
    "SE_Central":    ("sweden/south_central_sweden",   "SE-SE3"),
    "SE_South":      ("sweden/south_sweden",           "SE-SE4"),
    "Belgium":       ("belgium",                       "BE"),
    "Poland":        ("poland",                        "PL"),
    # ── Europe South (18) ────────────────────────────────────────────
    "Austria":       ("austria",                       "AT"),
    "Czechia":       ("czechia",                       "CZ"),
    "France":        ("france",                        "FR"),
    "Germany":       ("germany",                       "DE"),
    "Hungary":       ("hungary",                       "HU"),
    "IT_North":      ("italy/north_italy",             "IT-NO"),
    "IT_CentralNorth": ("italy/central_north_italy",   "IT-CNO"),
    "IT_CentralSouth": ("italy/central_south_italy",   "IT-CSO"),
    "IT_South":      ("italy/south_italy",             "IT-SO"),
    "IT_Sardinia":   ("italy/sardinia",                "IT-SAR"),
    "IT_Sicily":     ("italy/sicily",                  "IT-SIC"),
    "Portugal":      ("portugal",                      "PT"),
    "Serbia":        ("serbia",                        "RS"),
    "Slovakia":      ("slovakia",                      "SK"),
    "Slovenia":      ("slovenia",                      "SI"),
    "Spain":         ("spain",                         "ES"),
    "Switzerland":   ("switzerland",                   "CH"),
    "Croatia":       ("crotia",                        "HR"),
    # ── US East (8) ──────────────────────────────────────────────────
    "US_ISONE":      ("USA/iso_new_england_inc",                                    "US-NE-ISNE"),
    "US_NYISO":      ("USA/new_york_independent_system_operator",                   "US-NY-NYIS"),
    "US_PJM":        ("USA/pjm_interconnection_LLC",                                "US-MIDA-PJM"),
    "US_MISO":       ("USA/midcontinent_independent_transmission_system_operator",   "US-MIDW-MISO"),
    "US_DukeCarolinas":  ("USA/duke_energy_carolinas",                              "US-CAR-DUK"),
    "US_DukeProgressE":  ("USA/duke_energy_progress_east",                          "US-CAR-CPLE"),
    "US_DukeProgressW":  ("USA/duke_energy_progress_west",                          "US-CAR-CPLW"),
    "US_SCEG":       ("USA/south_carolina_electric_gas_company",                    "US-CAR-SCEG"),
    # ── US South (8) ─────────────────────────────────────────────────
    "US_Southern":   ("USA/southern_company_services_inc_trans",                     "US-SE-SOCO"),
    "US_Tallahassee": ("USA/city_of_tallahassee",                                   "US-FLA-TAL"),
    "US_FLMunicipal": ("USA/florida_municipal_power_pool",                          "US-FLA-FMPP"),
    "US_FPL":        ("USA/florida_power_and_light_company",                        "US-FLA-FPL"),
    "US_Gainesville": ("USA/gainesville_regional_utilities",                         "US-FLA-GVL"),
    "US_TVA":        ("USA/tennessee_valley_authority",                              "US-TEN-TVA"),
    "US_ERCOT":      ("USA/electric_reliability_council_of_texas",                   "US-TEX-ERCO"),
    "US_SPP":        ("USA/southwest_power_pool",                                    "US-CENT-SWPP"),
    # ── US West (15) ─────────────────────────────────────────────────
    "US_APS":        ("USA/arizona_public_service_company",                          "US-SW-AZPS"),
    "US_Avista":     ("USA/avista_corporation",                                      "US-NW-AVA"),
    "US_BANC":       ("USA/balancing_authority_of_northern_california",               "US-CAL-BANC"),
    "US_BPA":        ("USA/bonneville_power_administration",                         "US-NW-BPAT"),
    "US_CAISO":      ("USA/california_independent_system_operator",                  "US-CAL-CISO"),
    "US_ElPaso":     ("USA/el_paso_electric_company",                                "US-SW-EPE"),
    "US_IdahoPower": ("USA/idaho_power_company",                                     "US-NW-IPCO"),
    "US_LADWP":      ("USA/los_angeles_dept_of_water_and_power",                     "US-CAL-LDWP"),
    "US_NWEnergy":   ("USA/northwestern_energy",                                     "US-NW-NWMT"),
    "US_PacifiCorpE": ("USA/pacificorp_east",                                        "US-NW-PACE"),
    "US_PGE":        ("USA/portland_general_electric_company",                        "US-NW-PGE"),
    "US_PSCo":       ("USA/public_service_company_of_colorado",                      "US-SW-PNM"),
    "US_Turlock":    ("USA/turlock_irrigation_district",                              "US-CAL-TIDC"),
    "US_WAPA":       ("USA/western_area_power_administration_rocky_mountain_region",  "US-NW-WACM"),
    "US_AssocElec":  ("USA/associated_electric_cooperative",                          "US-MIDW-AECI"),
    # ── Middle East (7) ──────────────────────────────────────────────
    "Israel":        ("israel",                        "IL"),
    "Turkey":        ("turkey",                        "TR"),
    "Cyprus":        ("cyprus",                        "CY"),
    "Greece":        ("greece",                        "GR"),
    "Bulgaria":      ("bulgaria",                      "BG"),
    "Romania":       ("romania",                       "RO"),
    "SouthAfrica":   ("south_africa",                  "ZA"),
    # ── Asia (24) ────────────────────────────────────────────────────
    "AU_NSW":        ("australia/new_south_wales",     "AU-NSW"),
    "AU_NT":         ("australia/nothern_territory",   "AU-NT"),
    "AU_QLD":        ("australia/queensland",          "AU-QLD"),
    "AU_VIC":        ("australia/victoria",            "AU-VIC"),
    "IN_East":       ("india/eastern_india",           "IN-EA"),
    "IN_NorthEast":  ("india/north_eastern_india",     "IN-NE"),
    "IN_North":      ("india/northern_india",          "IN-NO"),
    "IN_South":      ("india/southern_india",          "IN-SO"),
    "IN_West":       ("india/western_india",           "IN-WE"),
    "JP_Chubu":      ("japan/chubu",                   "JP-CB"),
    "JP_Chugoku":    ("japan/chugoku",                 "JP-CG"),
    "JP_Hokkaido":   ("japan/hokkaido",                "JP-HKD"),
    "JP_Hokuriku":   ("japan/hokuriku",                "JP-HR"),
    "JP_Kansai":     ("japan/kansai",                  "JP-KN"),
    "JP_Okinawa":    ("japan/okinawa",                 "JP-ON"),
    "JP_Tohoku":     ("japan/tohoku",                  "JP-TH"),
    "JP_Tokyo":      ("japan/tokyo",                   "JP-TK"),
    "PH_Luzon":      ("philippines/luzon",             "PH-LU"),
    "PH_Mindanao":   ("philippines/mindanao",          "PH-MI"),
    "PH_Visayas":    ("philippines/visayas",           "PH-VI"),
    "Singapore":     ("singapore",                     "SG"),
    "SouthKorea":    ("south_korea",                   "KR"),
    "Taiwan":        ("taiwan",                        "TW"),
    "NewZealand":    ("new_zeland",                    "NZ"),
    # ── South America (7) ────────────────────────────────────────────
    "BR_Central":    ("brazil/central_brazil",         "BR-CS"),
    "BR_North":      ("brazil/north_brazil",           "BR-N"),
    "BR_NorthEast":  ("brazil/north_east_brazil",      "BR-NE"),
    "BR_South":      ("brazil/south_brazil",           "BR-S"),
    "Chile":         ("chile",                         "CL-SEN"),
    "Peru":          ("peru",                          "PE"),
    "Nicaragua":     ("nicaragua",                     "NI"),
}


class CarbonLoader:
    """Load and query hourly carbon intensity data.

    Usage:
        loader = CarbonLoader("/path/to/carbon_data", year=2024)
        ci = loader.get_ci("France", datetime(2024, 7, 15, 14, 0, 0))
        # Returns e.g. 45.2 gCO2eq/kWh
    """

    def __init__(
        self,
        carbon_data_root: str,
        year: int = 2024,
        regions: Optional[List[str]] = None,
    ) -> None:
        self.year = year
        # _data[region_name][(month, day, hour)] -> ci_value
        self._data: Dict[str, Dict[Tuple[int, int, int], float]] = {}

        regions_to_load = regions or list(REGION_CSV_MAP.keys())
        for region in regions_to_load:
            if region not in REGION_CSV_MAP:
                raise ValueError(f"Unknown region: {region}")
            subdir, zone_prefix = REGION_CSV_MAP[region]
            filename = f"{zone_prefix}_{year}_hourly.csv"
            filepath = os.path.join(carbon_data_root, subdir, filename)

            hourly_map: Dict[Tuple[int, int, int], float] = {}
            with open(filepath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dt_str = row["Datetime (UTC)"]
                    ci_str = row["Carbon Intensity gCO\u2082eq/kWh (direct)"]
                    if not ci_str:
                        continue
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    hourly_map[(dt.month, dt.day, dt.hour)] = float(ci_str)
            self._data[region] = hourly_map

    def get_ci(self, region: str, dt: datetime) -> float:
        """Return carbon intensity in gCO2eq/kWh for region at given UTC time."""
        hourly = self._data.get(region)
        if hourly is None:
            raise ValueError(f"No data loaded for region: {region}")
        key = (dt.month, dt.day, dt.hour)
        ci = hourly.get(key)
        if ci is not None:
            return ci
        # Fallback: average of all hours on that day
        day_vals = [v for (m, d, h), v in hourly.items()
                    if m == dt.month and d == dt.day]
        if day_vals:
            return sum(day_vals) / len(day_vals)
        # Last resort: average of entire month
        month_vals = [v for (m, d, h), v in hourly.items()
                      if m == dt.month]
        if month_vals:
            return sum(month_vals) / len(month_vals)
        raise ValueError(f"No CI data for {region} at {dt}")

    def get_all_regions(self) -> List[str]:
        return list(self._data.keys())

    def summary(self) -> str:
        lines = [f"CarbonLoader ({self.year}):"]
        for region, hourly in self._data.items():
            vals = list(hourly.values())
            lines.append(
                f"  {region:20s}: min={min(vals):6.1f}  "
                f"avg={sum(vals)/len(vals):6.1f}  "
                f"max={max(vals):6.1f} gCO2/kWh  ({len(vals)} hours)"
            )
        return "\n".join(lines)
