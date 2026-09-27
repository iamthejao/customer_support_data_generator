# Norrholt Glass Machinery — Company Profile (Seed File)

> **Purpose of this file.** A structured seed for generation software. It describes the fictional
> manufacturer **Norrholt Glass Machinery**: forming machines, feeders, controls and inspection
> equipment for container-glass plants (bottles and jars). Every company, product and person named
> here is invented; all data associated with Norrholt is fake/simulated.
>
> Fields marked **[inferred]** are enrichments a generator plausibly needs to populate a believable
> instance. They are consistent with the rest of this profile.

---

## Identity

**Company name.** Norrholt Glass Machinery
**Type.** Privately held capital-equipment manufacturer (fictional)
**Sector.** Machine manufacturing — glass-container forming and inspection
**Headquarters [inferred].** Gothenburg, Sweden; service hubs in Europe, the Americas and Asia.
**Tagline [inferred].** Every section, every gob, every bottle.

## Products at a glance

A container-glass plant melts glass in a furnace, then the hot end turns it into bottles: the
**feeder** cuts the molten stream into gobs, the **forming machine** blows each gob into a bottle
in two moulds (blank side, then blow side), and inspection rejects defective ware. Norrholt makes:

- **IS forming machines (FX-608, FX-612)** — individual-section machines with 8 or 12 sections,
  each section forming bottles independently; servo-driven invert, takeout and pusher mechanisms.
- **Gob feeders (GF-350)** — servo plunger, tube and shears that set gob weight and shape.
- **Forming controls (TC-400)** — electronic section timing: every mechanism's on/off angle per
  section, job recipes, alarms and event logs.
- **Ware inspection (VQ-220)** — camera and sensor inspection that rejects cracked, misshapen or
  thin-walled containers and reports defects per mould cavity.

Plants run 24/7. Job changes (a new bottle shape) happen every few days and are the riskiest
moment: new moulds, new timing recipe, new gob weight. Controls software is **SectionView 5.2**.

---

## What typically goes wrong [inferred]

- **Forming defects:** cracks ("checks") at the finish or base, thin walls, uneven glass
  distribution, choked necks, stuck ware. Causes include gob weight or temperature drift, mould
  temperature and cooling air, swabbing (mould lubrication), and timing settings.
- **Mechanisms:** invert or takeout servo faults, worn plungers, misaligned shears, pusher timing,
  leaking cooling-air valves.
- **Controls:** a timing recipe loaded from the wrong job, drift after a firmware update, a failed
  I/O module, network faults between the TC-400 cabinet and sections.
- **Inspection:** rising false rejects after a lamp or camera ages, or real defects passing because
  a sensor was mis-set after a job change.

---

## Service organisation

**Customer Service Desk (L1).** 24/7 hotline and email, staffed by service coordinators: spare
parts, documentation, recipe and software questions, known alarm codes, and routing.

**Process and controls engineers (L2).** Forming-process specialists who read TC-400 event logs
and timing recipes remotely, compare them with job records, and work through defect analyses with
the plant's hot-end team.

**Field service.** Service engineers who travel to plants for mechanism rebuilds, servo and
control-cabinet repairs and job-change support. The Showcase names **Ingrid Sollberg** for the
Nordics and Central Europe.

**Spare-part hubs [inferred].** Gothenburg central hub plus regional hubs; servo drives and
plungers are held for 24-hour dispatch in Europe.

---

## Systems landscape [inferred]

- **CRM** — accounts, installed base per plant and line, service contracts, interaction log.
- **ERP** — spare-part variants per machine generation, stock per hub, repair orders.
- **Remote service gateway** — secure access to TC-400 event logs and timing recipes.
- **Knowledge base** — troubleshooting by alarm code and by defect type.
- **Field-service scheduling** — engineer calendars, visit booking and travel.
- **Document repository** — manuals, mechanism drawings and parts lists by machine serial.

---

## Customers and how they engage

Container-glass plants making beverage bottles, food jars and pharmaceutical glass. A forming line
produces several hundred bottles a minute, so a defect wave or a stopped section costs money by the
minute. Callers are hot-end staff describing what they see on the line ("section 5 is throwing
checks on the finish") and often don't have the machine serial to hand.

---

## Case facts

Structured values the generator draws each case's facts from (the machine, the customer site and
the caller's job role), so both sides of a conversation talk about the same machine. Serial formats
use `#` for a digit and `?` for an upper-case letter.

### Assets

| Model | Description | Serial format |
|---|---|---|
| FX-608 | 8-section IS forming machine, double gob | FX608-####-?? |
| FX-612 | 12-section IS forming machine, triple gob | FX612-####-?? |
| GF-350 | Servo gob feeder with plunger and shears | GF350-####-?? |
| TC-400 | Electronic section timing and forming control system | TC400-#####-? |
| VQ-220 | Ware inspection machine | VQ220-####-?? |

### Caller roles

- hot-end supervisor
- forming shift supervisor
- IS machine operator
- maintenance engineer
- controls engineer
- production manager

### Site locales

- en_US
- en_GB
- de_DE
- fr_FR
- it_IT
- es_ES
- pl_PL
- cs_CZ
- es_MX

---

## Quick-reference fact sheet

| Field | Value |
|---|---|
| Company | Norrholt Glass Machinery |
| Status | Fictional running example for this project |
| Industry | Container-glass forming machines, feeders, controls and inspection |
| Customers | Container-glass plants (beverage bottles, food jars, pharmaceutical glass) |
| Machine families | IS forming machines, gob feeders, forming controls, ware inspection |
| Example firmware | TC-400 controller firmware v7.1.3 |
| Software | SectionView 5.2 (timing and recipes) |
| Service model | 24/7 L1 service desk, L2 process and controls engineers, field service |
| Named engineer | Ingrid Sollberg (Nordics and Central Europe) |
| Named hub | Gothenburg central hub (24-hour dispatch in Europe) |
| Knowledge governance [inferred] | Head of process engineering approves KB articles |

**Glossary.** IS machine — individual-section forming machine. Gob — the measured piece of molten
glass cut for one container. Blank side / blow side — the two moulds a container is formed in.
Finish — the neck and opening of a bottle. Checks — small cracks. Swabbing — lubricating the
moulds. Hot end — the forming area of the plant. Job change — switching the line to a new
container. SN — Serial Number.
