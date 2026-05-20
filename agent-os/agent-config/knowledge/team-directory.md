# Team Directory

Authoritative team structure for the organization. Used to validate team/department references and resolve aliases.

---

## Departments

### Engineering (`ENG`)
- **Code**: `ENG`
- **Headcount**: 45 FTE
- **Annual Budget**: $5,000,000
- **Department Head**: VP of Engineering
- **Cost Center Root**: 100

#### Engineering Teams
| Team | Code | HC | Description |
|---|---|---|---|
| Frontend | ENG-FE | 12 | Web app, design system |
| Backend | ENG-BE | 15 | Services, APIs |
| Infrastructure | ENG-INFRA | 8 | Platform, DevOps, SRE |
| Data Platform | ENG-DATA | 6 | Pipelines, warehouse |
| QA | ENG-QA | 4 | Test automation |

---

### Sales (`SALES`)
- **Code**: `SALES`
- **Headcount**: 30 FTE
- **Annual Budget**: $3,000,000
- **Department Head**: VP of Sales
- **Cost Center Root**: 200

#### Sales Teams
| Team | Code | HC | Description |
|---|---|---|---|
| Enterprise | SALES-ENT | 8 | Accounts > $100k ARR |
| Mid-Market | SALES-MM | 12 | Accounts $10k-100k ARR |
| SMB | SALES-SMB | 6 | Accounts < $10k ARR |
| Sales Ops | SALES-OPS | 4 | Tooling, reporting, ops |

---

### Marketing (`MKTG`)
- **Code**: `MKTG`
- **Headcount**: 20 FTE
- **Annual Budget**: $2,000,000
- **Department Head**: VP of Marketing
- **Cost Center Root**: 300

#### Marketing Teams
| Team | Code | HC | Description |
|---|---|---|---|
| Demand Gen | MKTG-DG | 8 | Pipeline generation, paid |
| Brand | MKTG-BRAND | 5 | Brand, design, events |
| Content | MKTG-CONTENT | 4 | Content, SEO, docs |
| Marketing Ops | MKTG-OPS | 3 | Tools, attribution |

---

### Operations (`OPS`)
- **Code**: `OPS`
- **Headcount**: 25 FTE
- **Annual Budget**: $2,500,000
- **Department Head**: COO
- **Cost Center Root**: 400

#### Ops Teams
| Team | Code | HC | Description |
|---|---|---|---|
| Customer Success | OPS-CS | 10 | Renewals, account growth |
| Support | OPS-SUPPORT | 8 | Tickets, technical support |
| Facilities | OPS-FAC | 4 | Office, equipment |
| IT | OPS-IT | 3 | Internal tooling |

---

### Finance (`FIN`)
- **Code**: `FIN`
- **Headcount**: 10 FTE
- **Annual Budget**: $1,000,000
- **Department Head**: CFO
- **Cost Center Root**: 501

#### Finance Teams
| Team | Code | HC | Description |
|---|---|---|---|
| Accounting | FIN-ACC | 5 | AP, AR, GL |
| FP&A | FIN-FPA | 3 | Planning, analysis |
| Treasury | FIN-TREAS | 2 | Cash management |

---

### HR (`HR`)
- **Code**: `HR`
- **Headcount**: 6 FTE
- **Annual Budget**: $600,000
- **Department Head**: CHRO
- **Cost Center Root**: 502

---

### Legal (`LEGAL`)
- **Code**: `LEGAL`
- **Headcount**: 4 FTE
- **Annual Budget**: $400,000
- **Department Head**: General Counsel
- **Cost Center Root**: 503

---

## Aliases (Department-level)

The Understanding Agent should resolve any of these to the canonical 3-5 letter code:

| Input | Resolves To |
|---|---|
| `engineering`, `eng`, `r&d`, `rd`, `tech`, `engr` | `ENG` |
| `sales`, `revenue team`, `gtm`, `go to market` | `SALES` |
| `marketing`, `mktg`, `mkt`, `growth` | `MKTG` |
| `operations`, `ops`, `cx`, `customer ops` | `OPS` |
| `finance`, `fin`, `accounting`, `acct` | `FIN` |
| `hr`, `human resources`, `people`, `people ops` | `HR` |
| `legal`, `compliance`, `legal & compliance` | `LEGAL` |

---

## Total Company Snapshot

| Metric | Value |
|---|---|
| Total Headcount | 140 FTE |
| Total Annual Budget | $14,500,000 |
| Departments | 7 |
| Cost Centers (leaf) | 18 |

---

## Validation Rules

When an uploaded file contains a `team` or `department` value:
1. Try exact match against canonical codes (`ENG`, `SALES`, etc.)
2. Try alias match (case-insensitive)
3. Try fuzzy match (Levenshtein distance ≤ 2) — but flag for review
4. If still no match: flag as `critical` anomaly, do not silently assign

When a file contains a `cost_center` value:
1. Must match a known 3-digit code under one of the roots above
2. Cost centers from departments shown here are valid; others are rejected
