# Kalvora Dental — Company Profile (Seed File)

> **Purpose of this file.** A structured seed for generation software. It describes the fictional
> manufacturer **Kalvora Dental**: dental-laboratory and chairside equipment plus the restorative
> materials processed on it. Every company, product and person named here is invented; all data
> associated with Kalvora is fake/simulated.
>
> Fields marked **[inferred]** are enrichments a generator plausibly needs to populate a believable
> instance. They are consistent with the rest of this profile.

---

## Identity

**Company name.** Kalvora Dental
**Type.** Privately held manufacturer of dental equipment and restorative materials (fictional)
**Sector.** Medical devices — dental laboratory and dental practice equipment, dental materials
**Headquarters [inferred].** St. Gallen, Switzerland; sales and service subsidiaries across Europe
and North America.
**Tagline [inferred].** Restorations that fit, first time.

## Products at a glance

Kalvora sells systems: a material, the machine that processes it, and the programs that connect the
two. Most service contacts involve one of these machines:

- **Firing furnaces (CF-600)** — fire and glaze ceramic veneers and crowns on programmable firing
  cycles (pre-dry, heat rate, holding temperature, vacuum, cooling).
- **Press furnaces (CP-800)** — press glass-ceramic ingots into investment moulds, and fire.
- **Sintering furnaces (SX-1500)** — sinter milled zirconia frameworks and crowns at up to 1,600 °C.
- **Milling units (MV-450, MV-650)** — wet and dry mill blocks and discs from CAD designs.
- **Curing lights (LQ-220)** — cordless LED lights that polymerise composite fillings chairside.

Materials [inferred]: **Vitrana LS** lithium-disilicate glass-ceramic (press ingots and CAD
blocks), **Zirvelle** multilayer zirconia discs, and **Kalvora Flow** composites. Each material has
validated firing, pressing or sintering programs stored in the furnaces; programs and firmware are
updated through the **Kalvora Connect** portal. CAD/CAM milling runs on **KalvoCAM** software.

---

## What typically goes wrong [inferred]

- **Furnaces:** firing results off (restorations too glossy, too opaque, cracked, bubbles), usually
  from a drifted temperature calibration, a worn muffle or thermocouple, a vacuum leak (door seal,
  hose, pump), the wrong program for the material, or overloaded firing trays.
- **Milling units:** broken or worn tools, chipped margins, poor fit from a skipped calibration,
  clogged water filter or low coolant additive, spindle or door-sensor errors, CAM export problems.
- **Curing lights:** falling light output from a damaged or dirty light guide, an aging battery, or
  a lens coated with composite, which gives under-cured restorations.
- **Materials:** shade mismatches, fractures or poor fit traced back to handling: wrong program,
  contaminated investment, wrong sintering speed for the disc layer, lot-specific instructions.

---

## Service organisation

**Customer Service Center (L1).** Dental technicians by training, reached by phone and email. They
handle documentation, program and firmware questions, known troubleshooting patterns, and route
repairs. They see the customer's installed base in CRM.

**Technical Service (L2).** Device specialists with remote diagnostics: they read the furnace's
event log and calibration history from Kalvora Connect, request photos of results, and run
calibration and vacuum tests with the customer.

**Application specialists [inferred].** Materials experts who analyse restoration failures
(fractures, shade, fit) that trace back to processing rather than to a device fault.

**Field service.** Regional technicians who replace muffles, heating elements, vacuum pumps,
spindles and control boards on site. The Showcase names **Lukas Brenner** for the Switzerland East
region.

**Repair centre and depots [inferred].** Curing lights and handpieces go to the St. Gallen repair
centre on a swap basis; spare parts ship from the St. Gallen central depot (next business day in
the EU).

---

## Systems landscape [inferred]

- **CRM** — accounts, installed base, warranty and service contracts, interaction log.
- **ERP** — spare-part variants, stock per depot, swap-unit pool, repair orders.
- **Kalvora Connect** — device registration, firmware and program updates, remote event logs.
- **Knowledge base** — troubleshooting articles keyed to device and error code.
- **Field-service scheduling** — technician calendars and visit booking.
- **Document repository** — operating manuals and processing instructions by device and material.

---

## Customers and how they engage

Dental laboratories (from two-person labs to milling centres) and dental practices with chairside
CAD/CAM. A lab's furnace or milling unit sits in the middle of the day's production: a failed
firing means remakes and missed delivery dates to dentists, so labs call when a batch comes out
wrong. Callers often describe the result ("the crowns came out milky") before the machine, and
often don't have the serial number to hand.

---

## Case facts

Structured values the generator draws each case's facts from (the machine, the customer site and
the caller's job role), so both sides of a conversation talk about the same machine. Serial formats
use `#` for a digit and `?` for an upper-case letter.

### Assets

| Model | Description | Serial format |
|---|---|---|
| CF-600 | Ceramic firing furnace | CF600-####-?? |
| CP-800 | Combined press and firing furnace | CP800-####-?? |
| SX-1500 | High-temperature zirconia sintering furnace | SX1500-###-?? |
| MV-450 | Four-axis wet milling unit for glass-ceramic blocks | MV450-####-?? |
| MV-650 | Five-axis wet and dry milling unit for blocks and discs | MV650-####-?? |
| LQ-220 | Cordless LED curing light | LQ220-#####-? |

### Caller roles

- dental technician
- dental laboratory manager
- CAD/CAM technician
- dentist
- dental assistant
- practice manager

### Site locales

- de_CH
- fr_CH
- de_DE
- de_AT
- it_IT
- en_GB
- en_US

---

## Quick-reference fact sheet

| Field | Value |
|---|---|
| Company | Kalvora Dental |
| Status | Fictional running example for this project |
| Industry | Dental laboratory and practice equipment; restorative materials |
| Customers | Dental laboratories, milling centres, dental practices |
| Device families | Firing, press and sintering furnaces; milling units; LED curing lights |
| Example firmware | Furnace firmware v3.4.2 |
| Software | KalvoCAM 6.1 (CAM), Kalvora Connect (device portal) |
| Service model | L1 customer service, L2 technical service, application specialists, field service |
| Named technician | Lukas Brenner (Switzerland East region) |
| Named depot | St. Gallen central depot (next business day, EU) |
| Knowledge governance [inferred] | Technical Service lead approves KB articles |

**Glossary.** CRM — Customer Relationship Management. ERP — Enterprise Resource Planning.
KB — Knowledge Base. L1 / L2 — first- and second-level support tiers. CAD/CAM — computer-aided
design and manufacturing. Muffle — the heated firing chamber of a furnace. SN — Serial Number.
