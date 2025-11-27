# Testing Checklist

Run through this list before presenting, committing, or deploying.

## Backend
- [ ] `uvicorn backend.api.main:app --reload` starts without tracebacks.
- [ ] Startup log says **"Loaded 2-bin classifier"** (or explicitly notes the regression fallback).
- [ ] `curl http://localhost:8000/health` returns `{"status":"ok"}`.
- [ ] `curl -X POST http://localhost:8000/predict -d '{"text":"sample","task_type":"classification"}'` returns JSON with `predicted_year_range` and `confidence`.

## Frontend
- [ ] `cd frontend && npm run dev` compiles with zero TypeScript/Tailwind warnings.
- [ ] `NEXT_PUBLIC_API_URL` points to the running backend.
- [ ] Hero badges read “RoBERTa-base · 2-bin classifier” and “Regression fallback always on”.
- [ ] Demo textarea produces a prediction + confidence bars when hitting Predict.
- [ ] Visualizations section either shows the PNGs or the placeholder instructions (no broken images).

## Content sanity
- [ ] Accuracy numbers read 85% / 85.3% everywhere (hero, overview, results, conclusion).
- [ ] Methodology cards mention RoBERTa, 2-bin setup, regression fallback, and FastAPI deployment.
- [ ] Footer displays “MAIS Machine Learning Bootcamp Fall 2025” and the McGill logo.
- [ ] Contact link points to `royelia43@gmail.com`.

## Assets
- [ ] `frontend/public/mcgill-logo.png` exists.
- [ ] Visualization PNGs exist or you’re okay with placeholders.
- [ ] Favicon (`frontend/public/favicon.png`) loads in the browser tab.

## Optional stretch checks
- [ ] Run `python generate_visualizations.py` after training to refresh charts.
- [ ] If classifier weights are missing, regression predictions land near 2016 (expected behavior) and the UI message explains the fallback.
- [ ] Test dark mode via the navbar toggle.

Print or bookmark this list so the whole team can verify quickly before a presentation.
