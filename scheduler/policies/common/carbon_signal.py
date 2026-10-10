"""The carbon price a placement decision sees for a datacentre.

A joule at the rack costs PUE joules at the meter, so a site's carbon per unit
of IT work is carbon_intensity x PUE, and that product -- not the grid's
intensity alone -- is what ranks sites. Before this the policies read the raw
intensity and PUE entered only the accounting: a site at PUE 1.08 on a 100 g
grid and one at PUE 1.32 on a 95 g grid were ranked the wrong way round, and
the reported grams disagreed with the choice that produced them.

The PUE is taken relative to PUE_REF, the uniform value every earlier run
used. A fleet at that value gets a factor of exactly 1, so those runs make
bit-identical decisions; with per-site values the factor is the ratio, and
every decision here is scale-free -- the greedy score normalises by the
candidates' maximum, the fallback is an argmin, and the LP's costs all scale
together -- so only the ratio between sites matters.
"""

#: The uniform PUE of every run before per-site values; see the module note.
PUE_REF = 1.1


def effective_carbon_intensity(stats) -> float:
    """carbon_intensity x PUE / PUE_REF; the raw intensity if PUE is unknown."""
    ci = float(getattr(stats, "carbon_intensity", 0.0) or 0.0)
    pue = float(getattr(stats, "pue", 0.0) or 0.0)
    return ci * pue / PUE_REF if pue > 0 else ci
