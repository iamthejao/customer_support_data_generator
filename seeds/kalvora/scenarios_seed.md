# Kalvora Dental — Customer Service Cases (Seed File)

> **Purpose of this file.** The customer-service scenarios for the fictional manufacturer
> **Kalvora Dental** (see `company_seed.md`). Each `##` section is one scenario: the category
> before the dash, the title after it. Phase 1 draws its problems from these. All names are
> invented; fields marked **[inferred]** are plausible enrichments.

---

## Documentation — Processing instructions for a new material

**Tier.** L1 customer service; no escalation expected.

**Initial customer request.**
> "We've just received our first Zirvelle multilayer discs. Can you send me the current
> processing instructions and the sintering program to use on our furnace?"

**Agent interaction.** Confirm the furnace model and serial number and the disc's shade and
height, check the furnace has the current program set, send the processing instructions revision
that matches, and point to the program update in Kalvora Connect if the furnace's set is older.

**What goes wrong today.** Instructions are revised per material generation; labs use an older
PDF or the wrong speed program, and the first sintered crowns come out too translucent or chalky.

**Sample-data hints [inferred].** Sintering furnace SX-1500; material "Zirvelle ML"; program set
"P-2025.2"; processing instructions revision "Rev. 04"; speed vs. standard sintering program.

---

## Firing results — Restorations come out too glossy or rounded

**Tier.** L1 customer service, known troubleshooting pattern; L2 if a calibration shows drift.

**Initial customer request.**
> "Since last week everything from our CF-600 comes out over-fired: the glaze is shiny like
> glass and the edges of the veneers are rounding off. Same programs we always use."

**Agent interaction (guided checklist).** Ask whether the program or material changed, how full
the firing tray was, and when the furnace was last temperature-calibrated; walk the customer
through the calibration test with the silver test sample; if it has drifted, adjust the
temperature offset, re-fire a test piece, and monitor.

**What goes wrong today.** Over-firing looks like a material problem, so labs remake crowns for
days before anyone checks the furnace's temperature calibration or its thermocouple.

**Sample-data hints [inferred].** Firing furnace CF-600; glaze firing at 770 °C; last calibration
"over 12 months ago"; calibration offset found +18 °C; silver test sample melts at 961 °C;
firmware v3.4.2.

---

## Press furnace — Vacuum errors and bubbles in pressed ceramics

**Tier.** L2 technical service with remote diagnostics.

**Initial customer request.**
> "Our CP-800 keeps stopping with a vacuum error halfway through pressing, and the pieces that
> do finish have little bubbles near the surface."

**Agent interaction.** Request the furnace event log from Kalvora Connect, ask for the vacuum
reached in the self-test, have the customer inspect and clean the door seal and the vacuum hose
connection, and run the vacuum test program; if the pump cannot hold vacuum, arrange a pump.

**What goes wrong today.** Vacuum faults have several causes (seal, hose, pump, blocked filter),
and labs replace the wrong part first.

**Sample-data hints [inferred].** Press furnace CP-800; error code "E-26 vacuum not reached";
vacuum self-test target below 50 mbar, measured 120 mbar; press ingot "Vitrana LS"; door seal
cracked at the hinge side.

---

## Milling — Chipped margins and poor fit after a tool change

**Tier.** L1 for tools and calibration; L2 for spindle faults.

**Initial customer request.**
> "Crowns from our MV-450 have chipped margins and they don't seat properly any more. It started
> after we changed the tools on Monday."

**Agent interaction.** Ask which tools were fitted and whether the tool life counter was reset,
check the water filter and coolant additive, and walk the customer through the calibration with
the calibration body; if the spindle runs rough, book a technician.

**What goes wrong today.** A skipped calibration after a tool change and a clogged filter look
the same from the outside; spindle bearing wear is found late.

**Sample-data hints [inferred].** Milling unit MV-450; KalvoCAM 6.1; tool set "Step bur 12S /
Cylinder pointed 12S"; calibration last run 5 weeks ago; water tank changed "every 2 weeks".

---

## Curing light — Composite not curing fully

**Tier.** L1 customer service; send-in swap through the repair centre.

**Initial customer request.**
> "Our fillings are staying soft at the bottom and the LQ-220 seems weaker than it used to be.
> The battery shows full."

**Agent interaction.** Ask the dentist or assistant to check the light guide for cracks or
cured composite, test the output on the charging base's radiometer, and compare with the
specification; if output stays low with a clean light guide, arrange a swap unit.

**What goes wrong today.** Practices keep using a light whose output has fallen, and the first
sign is failed fillings weeks later.

**Sample-data hints [inferred].** Curing light LQ-220; built-in radiometer on the base;
specified output 1,200 mW/cm²; measured 650 mW/cm²; light guide with a hairline crack; swap unit
shipped next business day.

---

## Field service — Sintering furnace heating fault

**Tier.** Field service with spare-part coordination.

**Initial customer request (urgent).**
> "Our SX-1500 stopped mid-cycle overnight with a heating error and the chamber is cold. We have
> twenty zirconia bridges that have to go out tomorrow."

**Agent interaction.** Confirm the error code and whether the cycle can be restarted, check the
event log for heating-element resistance values, reserve heating elements from the depot, and
book the regional technician; offer an interim sintering option at a partner lab if needed.

**What goes wrong today.** Coordinating the part, the technician and the customer's delivery
deadline takes hours of phone calls.

**Sample-data hints [inferred].** Sintering furnace SX-1500; error "E-41 heating circuit";
MoSi2 heating element set; technician Lukas Brenner (Switzerland East region); depot St. Gallen,
next business day; estimated repair 3 hours.
