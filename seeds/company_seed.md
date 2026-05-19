# CoolTherm Industrial Chillers — Company Profile (Seed File)

> **Purpose of this file.** A structured seed for generation software. It describes the fictional
> manufacturer **CoolTherm Industrial Chillers**, the running example used throughout this
> project so that customers, machines, and resolution flows stay concrete.
>
> Fields marked **[inferred]** are lightly enriched additions a generator would plausibly need to
> populate a believable instance. They are reasonable inferences consistent with the rest of this
> profile.

---

## Identity

CoolTherm Industrial Chillers is a fictional original equipment manufacturer (OEM) of industrial
water chillers. It exists only as a demonstration vehicle: this project uses it as a
running example so that prospects can see and feel how agentic AI customer service works against
concrete customers, machines, and resolution flows. All data associated with CoolTherm is
fake/simulated.

**Company name.** CoolTherm Industrial Chillers
**Type [inferred].** Privately held industrial-equipment manufacturer (fictional)
**Sector.** Machine manufacturing — industrial refrigeration / process cooling
**Tagline [inferred].** Mission-critical process temperature control.

---

## What CoolTherm makes

CoolTherm manufactures industrial water chillers — mission-critical equipment that regulates
process temperatures. The machines are sold into three primary end markets: plastics, food
processing, and chemical production. Because the chillers sit in the middle of a production line,
an outage stops the customer's production, which is what makes responsive service so valuable.

**Flagship model in the Showcase.** The **CT-500** is the unit referenced across the business
cases (documentation requests, temperature-performance troubleshooting, firmware faults, and
compressor failure all use the CT-500).

**Representative components [inferred].** Third-party compressors are used in the build — the
Showcase names a **Bitzer 4FES-5Y compressor** as a spare part. Other serviceable elements implied
by the cases include the condenser coil and fan assembly, the safety-interlock system, and onboard
firmware (e.g. firmware version v4.2.1).

**Product identifiers [inferred].** Each unit carries a nameplate with a model number and serial
number; manuals are versioned by revision. These identifiers are what the service organisation
needs in order to deliver the correct documentation and machine-specific guidance.

---

## Service organisation

CoolTherm operates a **global 24/7 service organisation**. It is structured in three tiers plus a
supporting logistics layer, and the five Showcase business cases trace this organisation end to
end — from L1 through L2 to field service and continuous learning.

**L1 — service-desk agents.** First-line support handling high-volume, repetitive requests
(documentation, known troubleshooting patterns). Resolution quality today varies by which agent
picks up the ticket.

**L2 — technical specialists with remote diagnostics.** Senior engineers who take cases the
knowledge base cannot match — typically after a firmware change or with rare configurations. They
request diagnostic logs, run remote sessions, and read firmware and configuration values. Their
time is expensive and scarce, and their expertise is the knowledge most at risk as they retire.

**Field-service network.** Regional technicians who handle everything physical — on-site repairs
and part replacements. The Showcase names **Martin Hofer** as a technician in the **Austria
region**.

**Regional spare-part warehouses.** The field-service network is supported by regional spare-part
warehouses. The Showcase names a **Vienna warehouse** with same-day availability for the compressor
spare.

**Knowledge function [inferred].** A knowledge manager role owns approval and publication of KB
articles, which is the human checkpoint in the continuous-learning case.

---

## Systems landscape [inferred]

The business cases imply CoolTherm runs a fairly standard manufacturer's back office, and the value
of agentic AI in the Showcase comes precisely from connecting these systems without "swivel-chair"
human effort:

- **CRM** — customer records, entitlement/warranty validation, interaction logging, follow-up queues, and capture of root causes as KB-article candidates.
- **ERP** — asset records for the installed base, part-variant identification, multi-warehouse inventory, and reservations.
- **Knowledge base (KB)** — symptom-to-pattern matching with confidence scores; draft, approval, and publication of articles.
- **Diagnostic-log tooling** — export and parsing of unit diagnostic logs for L2 root-cause analysis.
- **Field-service scheduling / dispatch** — technician availability and route-optimised scheduling.
- **Document repository** — versioned manuals keyed to model and revision.

---

## Customers and how they engage

CoolTherm's customers are industrial production sites in plastics, food processing, and chemical
manufacturing. They raise service requests by email or ticket, and they often omit machine
identifiers (model, serial number, revision), which today triggers manual back-and-forth before any
real work can start. Their defining characteristic is downtime sensitivity: when a chiller stops,
the customer's production line stops, so speed and accuracy of resolution translate directly into
the customer's own P&L.

**Customer profile [inferred].** Mid-to-large industrial plants operating continuously; maintenance
performed by the customer's own staff for L1-level physical actions (coil cleaning, inspections);
on-site OEM technicians called in for hardware failures.

---

## Strategic context — why CoolTherm is the right example

CoolTherm is built to embody two forces reshaping machine manufacturing. First, **aftermarket and
customer centricity**: service and parts are the highest-profit P&L line for a manufacturer like CoolTherm
(roughly 2× the gross margin of equipment sales), and customers increasingly buy the partnership,
not just the machine. Second, **the retirement of expertise**: CoolTherm's most valuable asset is
the judgement held by its L2 engineers, and that knowledge is walking out the door faster than it
can be replaced. A global 24/7 service organisation selling mission-critical equipment is exactly
where these two forces collide on every ticket — which is why CoolTherm makes the value of agentic
AI tangible.

---

## Quick-reference fact sheet

| Field | Value |
|---|---|
| Company | CoolTherm Industrial Chillers |
| Status | Fictional running example for this project |
| Industry | Industrial water chillers (process cooling) |
| End markets | Plastics, food processing, chemical production |
| Equipment role | Mission-critical process temperature regulation |
| Flagship model | CT-500 |
| Service model | Global 24/7 organisation: L1, L2, field service + regional warehouses |
| Named technician | Martin Hofer (Austria region) |
| Named warehouse | Vienna (same-day spare availability) |
| Named spare part | Bitzer 4FES-5Y compressor |
| Example firmware | v4.2.1 |
| Core systems [inferred] | CRM, ERP, KB, diagnostic logs, field-service scheduling, document repo |
| Knowledge governance [inferred] | Knowledge manager approves KB articles |
