# Demo Guide

This playbook walks through everything needed to run the live Reddit language drift demo.

## 1. Prep the environment
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cd frontend && npm install && cd ..`
4. Confirm `.env` (or shell) contains `NEXT_PUBLIC_API_URL=http://localhost:8000`

## 2. Ensure model weights exist
- Preferred experience: classifier weights inside `backend/models/reddit_year_model/` (config, tokenizer, `model.safetensors`).
- Regression fallback: keep `backend/models/reddit_year_model_regressor/` around so predictions still work if classifier weights are missing.

## 3. Start services
### Backend
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```
Watch the logs—on startup the API will say whether it loaded the classifier or fell back to regression.

### Frontend
```bash
cd frontend
NEXT_PUBLIC_API_URL="http://localhost:8000" npm run dev
```
Visit http://localhost:3000.

## 4. Demo flow
1. **Hero → Overview**: mention the 85% accuracy, 2-bin classifier, and regression fallback tags.
2. **Dataset / Methodology**: highlight the sampling strategy, RoBERTa fine-tuning, and FastAPI serving card.
3. **Interactive Demo**:
   - Type a recent-style sentence (`"no cap this meme is wild fr"`) and show it predicts 2020-2022.
   - Type an older-style sentence (`"Anyone else using IRC on XP?"`) to show 2008-2010.
   - Point to the confidence bars below the prediction.
4. **Visualizations**: walk through the confusion matrix and prediction vs actual plots if the PNGs are present. Note that placeholders appear if images are missing.
5. **Conclusion**: reiterate that the regression fallback ensures the endpoint is always live and outline next steps.

## 5. Reset between attendees
- Clear the textarea in the demo section.
- If you changed the theme (light/dark), reset to the preferred default.
- Keep the backend terminal visible so you can show logs if asked.

## 6. Troubleshooting quick hits
| Issue | Fix |
| --- | --- |
| `/predict` returns 500 | Check backend logs; usually means model weights are missing or the path is incorrect. |
| Frontend shows placeholder predictions | Ensure FastAPI is running on port 8000 and `NEXT_PUBLIC_API_URL` is set. |
| All predictions say 2016 | Backend is loading the regression model. Drop the classifier weights in `backend/models/reddit_year_model/` and restart. |
| Images missing in Visualizations | Place PNGs in `frontend/public/` with the expected filenames. |

You're ready to demo! Keep `docs/testing-checklist.md` nearby to verify everything before showtime.
