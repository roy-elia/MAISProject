---
owner: roy-elia
status: active
last_updated: 2026-11-24
source_of_truth: local
tags: [index]
---
# docs index

Purpose: shared, durable context for roy-elia/MAISProject. Read this before placing or finding a file.

| File | Summary (one line) | Read when | Owner |
|---|---|---|---|
| `COPY_MODEL_FILES.md` | How to copy trained model files into the backend model folders | Inference fails or weights are missing | roy-elia |
| `GENERATE_VISUALIZATIONS.md` | How to generate the figures shown on the website | Regenerating charts or word clouds | roy-elia |
| `deliverable-4-proposal.md` | Course deliverable: final demonstration proposal | Writing about project goals or the demo | roy-elia |
| `demo-guide.md` | Step-by-step walkthrough for the live demo | Preparing a demo | roy-elia |
| `testing-checklist.md` | Manual checklist before presenting, committing or deploying | Before presenting or deploying | roy-elia |

## Where does X go
| I am producing | Put it in | Owning skill | Tracked |
|---|---|---|---|
| Backend API code | `backend/api/` | manual | Yes |
| Model logic and data loading | `backend/core/` | manual | Yes |
| Training scripts | `backend/training/` | manual | Yes |
| Plot and word-cloud scripts | `backend/scripts/` | manual | Yes |
| Trained model weights | `backend/models/` (large; ask first) | manual | Yes (already tracked) |
| Website pages and components | `frontend/app/`, `frontend/components/` | manual | Yes |
| Website images and figures | `frontend/public/` | manual | Yes |
| Datasets | `data/` (see `data/README.md`) | manual | Yes (already tracked) |
| Startup and training shell scripts | `scripts/` | manual | Yes |
| Project guides | `docs/<topic>.md` with front matter | manual | Yes |
| A decision and its reasoning | `docs/decisions/YYYY-MM-DD-<topic>.md` (created on first use) | manual | Yes |
| Session handoff | `state/maisproject/HANDOFF.md` (created on first use) | session-handoff | Yes |
| Exploratory draft | `output/` | any | No |
| Scratch or pulled copy of an external page | `temp/` | any | No |

## Sources of truth
| Content | System of record | Role of this repo | Link recorded in |
|---|---|---|---|
| Code, docs, decisions | This repository | Owner record | none |
| Model results and accuracy figures | `README.md` (Results, Final Model) | Owner record | none |
| Deployed frontend and API | Vercel and Render (URLs in `README.md`) | Source code only | `README.md` |
| Raw Reddit data | Academic Torrents (see `data/README.md`) | Cleaned samples only | `data/README.md` |
| Tickets and bugs | GitHub Issues on roy-elia/MAISProject (assumed; see owner questions in `AGENTS.md`) | Link only | Pull request description |

## Rules for this folder
- Every Markdown file here starts with front matter (`templates/front-matter.md` in ai-standards).
- Summaries come before raw sources; raw files go in a `sources/` folder beside their summary.
