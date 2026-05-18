# CoolTherm — Customer Service Cases (Seed File)

> **Purpose of this file.** A structured seed for generation software. It describes the five
> customer-service business cases used in the Wissant Showcase, with the fictional manufacturer
> **CoolTherm Industrial Chillers** as the running example. Each case is written as a left-to-right
> story: what it costs today, how agentic AI changes it, and what the customer gets back.
>
> Fields marked **[inferred]** are lightly enriched additions a generator would plausibly need
> (systems touched, escalation tier, sample-data hints). They are reasonable inferences that extend
> the core business cases.

---

## How to read the case structure

Every case below follows the same shape so a generator can template against it:

- **Case ID & title** — short handle plus the descriptive name of the case.
- **Tags** — a one-line characterisation of the case (e.g. "high volume · repetitive").
- **Escalation tier [inferred]** — where the case sits in CoolTherm's service organisation (L1 / L2 / field service / cross-cutting).
- **Initial customer request** — the verbatim opening message from the customer.
- **Agent interaction** — the agentic AI's first move (clarifying question, checklist, log request, or confirmation).
- **As-is process & challenges** — how the work is done today and why it hurts.
- **Agentic-AI approach** — what the agent does end-to-end.
- **Systems touched [inferred]** — the back-office systems the agent reads from or writes to.
- **Human-in-the-loop [inferred]** — where a person stays in control.
- **How this helps the customer** — the business outcomes.
- **Sample-data hints [inferred]** — concrete values a generator can use to populate a believable instance.

---

## Case 01 — Documentation Request

**Title.** Documentation Request — high-volume, low-complexity, pure cost burden.

**Tags.** High volume · repetitive.

**Escalation tier [inferred].** L1 service desk; fully automatable, no escalation expected.

**Initial customer request.**
> "Hello, I need to schedule preventive maintenance next week and require the current maintenance
> manual for our industrial chiller. Could you please send me the relevant documentation?"

**Agent interaction (clarifying reply).**
> "Thank you for your request. To locate the correct documentation for your unit, could you please
> provide the model number and the serial number (found on the nameplate)?"

**As-is process & challenges.**
Customers raise service requests by email or ticket and often omit machine identifiers — model,
serial number, revision. Closing that gap requires manual email ping-pong. The L1 agent then
validates entitlement in CRM, looks up the asset record in ERP, finds the right manual revision,
and replies. It runs five to ten minutes per request, multiplied by thousands per year — and the
wrong revision still sometimes ships.

**Agentic-AI approach.**
The agent classifies the request, asks one focused clarifying question only if needed, validates the
customer and asset record automatically, retrieves the correct manual revision, and replies
end-to-end with no human time consumed. The interaction is logged in CRM and a 24-hour follow-up is
queued.

**Systems touched [inferred].** CRM (entitlement validation, interaction logging, follow-up queue);
ERP (asset record lookup); document/knowledge repository (manual revision retrieval).

**Human-in-the-loop [inferred].** None required for the standard path; the case is designed to
resolve without human time.

**How this helps the customer.**
Frees L1 capacity for cases that need human judgement. Delivers a faster, more accurate response —
the right manual revision is always validated against the installed-base record, in any time zone,
around the clock.

**Sample-data hints [inferred].** Model "CT-500"; serial number on nameplate; manual type
"preventive maintenance manual"; revision identifier (e.g. Rev. C); follow-up SLA "24 hours".

---

## Case 02 — L1 Remote Resolution

**Title.** L1 Remote Resolution — production-impacting, but a known troubleshooting pattern.

**Tags.** Production impact · known pattern.

**Escalation tier [inferred].** L1 service desk; resolved at first contact, no escalation to L2.

**Initial customer request.**
> "Our CT-500 is not reaching its target temperature. The setpoint is 12 °C but the unit is only
> achieving 18 °C even after running for 4 hours. The condenser fan appears to be running normally."

**Agent interaction (guided checklist).**
> "Likely root cause: condenser airflow obstruction. Please run: 1) inspect coil for dust/debris,
> 2) check 600 mm clearance, 3) verify fan rotation, 4) clean coil with compressed air (max 4 bar),
> 5) restart and monitor 30 min."

**As-is process & challenges.**
L1 agents work through troubleshooting from memory or scattered KB articles. Resolution quality
depends on which agent picks up the ticket. Production stays down longer than necessary, and the
fix is sometimes captured only in the agent's head — not in the asset history.

**Agentic-AI approach.**
The symptom pattern is matched to the knowledge base with a confidence score. A structured,
machine-specific guided checklist is sent to the customer. The customer's confirmation closes the
ticket, maintenance is logged in the asset history, and a follow-up is queued.

**Systems touched [inferred].** Knowledge base (symptom-to-pattern match with confidence score);
CRM / asset history (logging the maintenance action, follow-up queue); ERP (asset record for
machine-specific checklist parameters).

**Human-in-the-loop [inferred].** None on the matched path; low-confidence matches would escalate
to L2 (see Case 03). The physical inspection and cleaning are performed by the customer's own
maintenance staff.

**How this helps the customer.**
Higher first-contact resolution and reduced MTTR on the most common production-down patterns. A
consistent customer experience, independent of which agent picks up the ticket.

**Sample-data hints [inferred].** Model "CT-500"; setpoint 12 °C vs. actual 18 °C; run time 4 hours;
clearance spec 600 mm; compressed-air limit 4 bar; monitoring window 30 min; KB match confidence
score (e.g. 0.92).

---

## Case 03 — L2 Remote Resolution

**Title.** L2 Remote Resolution — complex root cause, expensive expert time.

**Tags.** Complex root cause · expert time.

**Escalation tier [inferred].** L2 technical specialists with remote diagnostics; reached when the
KB cannot match at L1.

**Initial customer request.**
> "Since our CT-500 was updated to firmware v4.2.1, the unit is triggering high-pressure faults
> every 2–3 hours. The unit shuts down, we reset it, it runs for a while, then faults again."

**Agent interaction (request for diagnostic logs).**
> "To enable remote diagnosis, please export the unit's diagnostic logs from the past 7 days
> (Menu → Diagnostics → Export Log → Last 7 days), save to USB, and reply with the file attached."

**As-is process & challenges.**
When the KB cannot match — typically after a firmware change or with rare configurations — the case
escalates to L2. A senior engineer requests diagnostic logs, sets up a remote session, reads
firmware and configuration values, identifies the root cause, and walks the customer through the
fix. One to three hours of expert time per case.

**Agentic-AI approach.**
On low-confidence cases, the customer is asked for a diagnostic-log export with the exact menu path.
The log is parsed, configuration drift or fault patterns are identified, and a corrected
configuration is generated and walked through with the customer. The root cause is captured in CRM
as a candidate for a new KB article (feeds Case 05).

**Systems touched [inferred].** Diagnostic-log parser; knowledge base (low-confidence match
trigger); CRM (root cause captured as KB-article candidate); configuration/firmware management
(generating the corrected configuration package).

**Human-in-the-loop [inferred].** L2 engineers stay in the loop for genuinely novel root causes;
only the "simpler L2 patterns" of this case are proposed for automation in an initial pilot.

**How this helps the customer.**
MTTR reduced from hours to minutes on already-seen complex patterns. L2 capacity is protected for
genuinely novel problems, with root causes captured systematically as seeds for future KB articles.

**Sample-data hints [inferred].** Model "CT-500"; firmware "v4.2.1"; fault type "high-pressure
fault"; fault interval "every 2–3 hours"; log export path "Menu → Diagnostics → Export Log → Last 7
days"; log window "7 days"; failure mode "post-update configuration drift".

---

## Case 04 — Spare Part Orchestration

**Title.** Spare Part Orchestration — urgent hardware failure, multi-system coordination.

**Tags.** Urgent · production down.

**Escalation tier [inferred].** Field service — regional technician network plus regional
spare-part warehouses; the agent orchestrates across systems but the repair itself is physical.

**Initial customer request (urgent).**
> "URGENT — Production down. Loud abnormal noise from the compressor; discharge pressure near zero.
> Unit shut down on safety interlock. We need parts and a technician on-site as soon as possible."

**Agent interaction (confirmation).**
> "Spare part reserved: Bitzer 4FES-5Y compressor — Vienna warehouse (same-day). Technician
> assigned: Martin Hofer, Austria region. Appointment: tomorrow at 08:00. Estimated repair:
> 4–6 hours. Please do not restart the unit."

**As-is process & challenges.**
A production-down hardware failure. A service coordinator is on the phone for hours: validating the
failure mode, identifying the right part variant, checking warranty, calling regional warehouses to
find inventory, creating an ERP reservation, calling technicians for availability, confirming the
slot, updating the customer. Any error in the chain extends downtime by days.

**Agentic-AI approach.**
The failure is validated, the exact part variant identified, warranty confirmed, and inventory
checked across regional warehouses in parallel. An ERP reservation is created, a certified
technician is scheduled with route optimisation, and the customer receives a structured
confirmation listing reserved part, technician, and arrival time. Hours of human coordination are
compressed into minutes — without sacrificing rigour.

**Systems touched [inferred].** CRM (warranty/entitlement check); ERP (part-variant identification,
multi-warehouse inventory check, reservation creation); field-service scheduling / dispatch system
(technician availability, route optimisation); customer notification channel (structured
confirmation).

**Human-in-the-loop [inferred].** Field technicians stay in the loop for everything physical;
spare-part decisions above defined value thresholds are escalated to a human.

**How this helps the customer.**
Minutes-to-decision and reduced production-down duration on the cases where every hour costs the
customer money. Better technician utilisation via route-optimised scheduling and consistent
customer communication every time.

**Sample-data hints [inferred].** Failure mode "compressor failure / discharge pressure near zero,
safety interlock"; part "Bitzer 4FES-5Y compressor"; warehouse "Vienna (same-day)"; technician
"Martin Hofer, Austria region"; appointment "tomorrow 08:00"; estimated repair "4–6 hours".

---

## Case 05 — Continuous Learning

**Title.** Continuous Learning — capturing institutional knowledge as it is generated.

**Tags.** Knowledge retention · compounding value.

**Escalation tier [inferred].** Cross-cutting — operates above L1/L2 on resolved-case data, with a
knowledge manager as approver.

**Initial customer request.** None — this case is triggered by patterns across resolved tickets
rather than a single inbound message.

**Agent interaction.** Internal: pattern detection and a draft KB article routed for approval (no
customer-facing message).

**As-is process & challenges.**
The same problem recurs across customers. L2 engineers handle each instance manually, and the
lesson stays in their heads or in tickets nobody re-reads. KB articles get written sporadically. As
senior engineers retire, the institutional knowledge they built leaves with them.

**Agentic-AI approach.**
Resolved cases are clustered to detect emerging patterns. When a recurring root cause is
identified, a draft KB article is generated, routed to a knowledge manager for approval, and
published. The next ticket matching that pattern then resolves automatically at L1.

**Systems touched [inferred].** CRM / ticket history (clustering resolved cases); knowledge base
(draft article generation and publication); approval workflow (routing to a knowledge manager).

**Human-in-the-loop [inferred].** A knowledge manager approves every KB article before publication.

**How this helps the customer.**
Knowledge is captured systematically as it is generated; recurring problems shift left from L2 to
L1, increasing FCR every quarter. Service capacity grows on its own as the knowledge base
compounds — even as senior engineers retire.

**Sample-data hints [inferred].** Pattern source "clustered L2 resolutions" (e.g. the CT-500
firmware v4.2.1 high-pressure fault from Case 03); article state machine "draft → approved →
published"; approver role "knowledge manager".

---

## Cross-case context

**The two forces that motivate all five cases.** First, aftermarket and customer centricity:
customers demand expertise, and aftermarket has become the highest-profit P&L line (roughly 2× the
gross margin of equipment sales; customer-centric manufacturers grow revenue ~2× vs. peers; 96% of
OEMs forecast parts-and-services growth over the next three years; yet only ~15% of manufacturers
consistently act on customer insights). Second, decades of expertise are walking out the door:
>10,000 Baby Boomers retire per day, up to 50% of skilled manufacturing openings could go unfilled,
new service engineers take 3–6 months to onboard, and ~23% of machine downtime is caused by human
error. Customer service is where these two forces collide on every ticket.

**Proposed pilot scope.** Start with Cases 1 and 2 plus the simpler L2 patterns from Case 3,
measured against today's baseline FCR, MTTR, and L2 escalation rate. Within one
quarter the impact is measurable; the path to Cases 4 and 5 then becomes a business decision, not a
technical one. Engineers stay in the loop for novel root causes, KB approvals, and spare-part
decisions above defined thresholds; field technicians stay in the loop for everything physical.
Agentic AI absorbs the swivel-chair work between systems — not engineering judgement.

**Glossary.** CRM — Customer Relationship Management. ERP — Enterprise Resource Planning.
KB — Knowledge Base. L1 / L2 — first- and second-level customer-support tiers. FCR — First-Contact
Resolution. MTTR — Mean Time To Resolution. OEM — Original Equipment Manufacturer. P&L — Profit and
Loss. SN / P/N — Serial Number / Part Number.

**Complexity ladder (summary).** Case 01 is high-volume, low-complexity cost burden. Case 02 is
production-impacting but a known pattern. Case 03 is a complex root cause consuming expensive expert
time. Case 04 is an urgent hardware failure requiring multi-system coordination. Case 05 is not a
single ticket at all but the learning loop that makes every future ticket cheaper.
