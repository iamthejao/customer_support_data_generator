# Norrholt Glass Machinery — Customer Service Cases (Seed File)

> **Purpose of this file.** The customer-service scenarios for the fictional manufacturer
> **Norrholt Glass Machinery** (see `company_seed.md`). Each `##` section is one scenario: the
> category before the dash, the title after it. Phase 1 draws its problems from these. All names
> are invented; fields marked **[inferred]** are plausible enrichments.

---

## Documentation — Parts list and drawings for a mechanism rebuild

**Tier.** L1 service desk; no escalation expected.

**Initial customer request.**
> "We're rebuilding the invert mechanisms on one of our IS machines during next week's shutdown.
> Can you send the current parts list and assembly drawings?"

**Agent interaction.** Confirm the machine model and serial number (mechanism generations
changed over the years), identify the invert variant fitted, send the matching parts list and
drawing revision, and offer a quote for the rebuild kit.

**What goes wrong today.** Parts lists are serial-specific; a list for the wrong generation
means the wrong seals and bearings arrive on shutdown day.

**Sample-data hints [inferred].** IS machine FX-612; invert "servo invert, generation 3";
drawing revision "D"; shutdown "next Tuesday"; rebuild kit quote within 24 hours.

---

## Forming defects — Checks on the finish in one section

**Tier.** L1 service desk with a known troubleshooting pattern; L2 if the pattern doesn't match.

**Initial customer request.**
> "Since this morning section 5 on line 2 is throwing checks on the finish, both cavities. The
> other sections are fine. The inspection machine is rejecting about 8% from that section."

**Agent interaction (guided checklist).** Ask what changed (job change, swabbing, mould set,
timing), have the team check the neck-ring and blank mould temperatures on that section, check
the cooling-air valve for the neck ring, and compare the section's timing with its neighbours;
adjust and monitor rejects for 30 minutes.

**What goes wrong today.** Hot-end staff change several settings at once; the cause is not
recorded and the defect returns on the next shift.

**Sample-data hints [inferred].** IS machine FX-608; section 5; reject rate 8% vs. 0.5% baseline;
neck-ring temperature 60 °C colder than neighbouring sections; neck-ring cooling valve stuck open.

---

## Controls — Timing drift after a firmware update

**Tier.** L2 process and controls engineers with remote access.

**Initial customer request.**
> "After the TC-400 firmware update on Friday, several sections are forming uneven wall
> thickness and the takeout sometimes misses the bottle. Nobody touched the timing."

**Agent interaction.** Request remote access or an export of the TC-400 event log and the active
job recipe, compare the section timing angles before and after the update, and walk the controls
engineer through correcting or reloading the recipe; monitor wall-thickness rejects.

**What goes wrong today.** Firmware updates occasionally convert timing recipes differently; the
plant sees process defects and blames glass temperature.

**Sample-data hints [inferred].** Forming controls TC-400; firmware v7.1.3; SectionView 5.2;
recipe "job 4471, 330 ml beer bottle"; takeout "on" angle shifted by 4 degrees; affected sections
3, 4 and 7.

---

## Feeder — Gob weight wandering

**Tier.** L2 process engineers; field service if the plunger mechanism is worn.

**Initial customer request.**
> "Gob weight on line 1 keeps drifting, plus or minus three grams, and we're getting light
> bottles and choked necks. The operators keep correcting the tube height."

**Agent interaction.** Ask for the gob-weight trend, check the plunger stroke and speed settings
and the tube height, ask whether the forehearth temperature is stable, and inspect the plunger tip
for wear; if the servo shows position errors, arrange a service engineer.

**What goes wrong today.** Weight drift can come from glass temperature, plunger wear or the
servo; operators chase it manually shift after shift.

**Sample-data hints [inferred].** Gob feeder GF-350; target gob weight 385 g; drift ±3 g;
forehearth temperature stable at 1,150 °C; plunger servo following error alarms at night.

---

## Inspection — False rejects after a job change

**Tier.** L1 service desk; L2 if settings don't explain it.

**Initial customer request.**
> "Since yesterday's job change the VQ-220 is rejecting around 12% for 'finish defects', but
> when we look at the rejected bottles, most of them are fine."

**Agent interaction.** Ask whether the inspection recipe was switched with the job, check the
lighting and camera cleanliness, compare reject images with good bottles, and walk the operator
through loading and verifying the correct inspection recipe; monitor the reject rate.

**What goes wrong today.** Good bottles are melted back down and throughput drops while the team
argues whether the defects are real.

**Sample-data hints [inferred].** Inspection machine VQ-220; reject rate 12% vs. 1% baseline;
inspection recipe still set to the previous 500 ml jar job; dirty finish camera lens.

---

## Field service — Invert servo fault stops a section

**Tier.** Field service with spare-part coordination.

**Initial customer request (urgent).**
> "Section 9 on our FX-612 has stopped with an invert servo fault and won't reset. We're running
> on eleven sections and losing a lot of bottles every hour."

**Agent interaction.** Confirm the alarm code and what resetting did, check the event log for
motor temperature and following errors, confirm whether the drive or the motor is at fault,
reserve the part from the hub, and book the service engineer.

**What goes wrong today.** Coordinating the part, the engineer and the plant's production
schedule takes hours; the wrong drive variant arrives.

**Sample-data hints [inferred].** IS machine FX-612; section 9; alarm "A-312 invert following
error"; servo drive variant "SD-40 rev B"; engineer Ingrid Sollberg; Gothenburg hub, 24-hour
dispatch; estimated repair 4 hours.
