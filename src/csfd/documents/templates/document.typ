// One template for every supporting document. It reads the document IR as JSON
// from sys.inputs.payload ({doc, files}) and lays it out like a corporate
// technical document. Values are inserted as data, never parsed as markup.
// Each section emits <secmark> metadata at its start and at the end of its own
// blocks, so csfd can map section ids to PDF pages (Compiler.query).
#let payload = json(bytes(sys.inputs.payload))
#let d = payload.doc
#let files = payload.files

#set document(
  title: if d.subtitle != none { d.title + " - " + d.subtitle } else { d.title },
  author: d.company,
  keywords: (d.doc_type, d.doc_number, ..d.applies_to.models),
  date: auto,
)
#set text(font: "Libertinus Serif", size: 10pt, lang: d.language)
#set par(justify: true)
#show heading.where(level: 1): set text(size: 14pt)
#show heading.where(level: 2): set text(size: 12pt)
#show heading.where(level: 3): set text(size: 11pt)

#let signal = (
  danger: ("DANGER", rgb("#C00000")),
  warning: ("WARNING", rgb("#E36C09")),
  caution: ("CAUTION", rgb("#BF8F00")),
  notice: ("NOTICE", rgb("#1F4E79")),
  note: ("NOTE", rgb("#595959")),
)

#set page(
  paper: "a4",
  margin: (x: 2cm, top: 2.5cm, bottom: 2.2cm),
  header: [
    #set text(size: 8pt)
    #d.company #h(1fr) #d.doc_number #h(1fr) Rev. #d.revision
    #line(length: 100%, stroke: 0.4pt)
  ],
  footer: context [
    #set text(size: 8pt)
    #line(length: 100%, stroke: 0.4pt)
    #d.title #h(1fr) Page #counter(page).display() of #counter(page).final().first()
  ],
)

#let grid-table(columns, rows) = table(
  columns: columns.len(),
  stroke: 0.5pt + luma(150),
  fill: (_, y) => if y == 0 { luma(230) },
  table.header(..columns.map(c => strong(c))),
  ..rows.flatten(),
)

#let render-block(b) = {
  if b.kind == "paragraph" {
    par(b.text)
  } else if b.kind == "admonition" {
    let (word, col) = signal.at(b.level)
    block(width: 100%, fill: luma(245), stroke: (left: 3pt + col), inset: 8pt, radius: 2pt)[
      #text(fill: col, weight: "bold")[#word] #h(0.6em) #b.text
    ]
  } else if b.kind == "procedure" {
    block(above: 1em, below: 0.6em, strong[Procedure: #b.title])
    enum(..b.steps.map(s => [#s.text #if s.expected != none { emph[Result: #s.expected] }]))
  } else if b.kind == "list" {
    list(..b.items.map(i => [#i]))
  } else if b.kind == "table" {
    [#figure(
      grid-table(b.columns, b.rows),
      caption: b.caption,
      kind: table,
      supplement: none,
      numbering: none,
    ) #label(b.id)]
  } else if b.kind == "figure" {
    [#figure(
      image(files.at(b.id), width: 85%, alt: b.alt),
      caption: b.caption,
      supplement: none,
      numbering: none,
    ) #label(b.id)]
  }
}

#let mark(id, edge) = context [#metadata((id: id, edge: edge, page: here().page())) <secmark>]

#let render-section(s, depth) = {
  [#heading(level: calc.min(depth, 3))[#s.number #s.title] #label(s.id)]
  mark(s.id, "start")
  for b in s.blocks { render-block(b) }
  mark(s.id, "end")
  for c in s.subsections { render-section(c, depth + 1) }
}

// Front matter: title, document control table, revision history, contents.
#align(center)[
  #v(1cm)
  #text(size: 20pt, weight: "bold")[#d.title]
  #if d.subtitle != none {
    v(0.2cm)
    text(size: 14pt)[#d.subtitle]
  }
  #v(0.6cm)
]
#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + luma(150),
  [Document number], d.doc_number,
  [Revision], d.revision,
  [Issue date], d.issue_date,
  [Status], d.status,
  [Applies to], {
    let all = d.applies_to.models + d.applies_to.serials
    if all.len() > 0 { all.join(", ") } else { "-" }
  },
  [Audience], d.audience,
)
#if d.revisions.len() > 0 {
  heading(level: 1, outlined: false)[Revision history]
  grid-table(
    ("Rev.", "Date", "Description", "Author"),
    d.revisions.map(r => (r.rev, r.issued, r.description, r.author)),
  )
}
#outline(title: [Contents], depth: 2)
#pagebreak()

#for s in d.sections { render-section(s, 1) }
