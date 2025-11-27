# How to Copy Your Trained Model Files

## Quick Steps

1. **Copy all your model files** to the `./reddit_year_model/` directory:

```bash
# If your files are in a folder (e.g., Downloads or a checkpoint folder):
cp /path/to/your/files/* ./reddit_year_model/

# Or if you're copying from a specific location:
# Example: if files are in ~/Downloads/model_files/
cp ~/Downloads/model_files/* ./reddit_year_model/
```

2. **Verify the critical file is there:**
```bash
ls -lh ./reddit_year_model/model.safetensors
```
You should see a large file (likely 100+ MB). This is your trained model weights.

3. **Check what's in the directory:**
```bash
ls -la ./reddit_year_model/
```

You should see:
- ✅ `model.safetensors` (the trained weights - REQUIRED)
- ✅ `config.json` (already there)
- ✅ `tokenizer.json` (already there)
- ✅ `tokenizer_config.json` (from your files)
- ✅ `vocab.json` (already there)
- ✅ `special_tokens_map.json` (already there)
- ✅ `merges.txt` (already there)
- Plus any optional training state files

4. **Restart the FastAPI backend** so it loads the new model:
```bash
# Stop the current backend (Ctrl+C if running)
# Then restart:
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Check the startup logs** - you should see:
```
Loaded 2-bin classifier model from ./backend/models/reddit_year_model
```

If you see that message, the model is loaded correctly! 🎉

## File Breakdown

| File | Purpose | Required? |
|------|---------|-----------|
| `model.safetensors` | **Trained model weights** | ✅ **YES - Critical!** |
| `tokenizer_config.json` | Tokenizer configuration | ✅ Yes |
| `config.json` | Model architecture config | ✅ Yes (already there) |
| `tokenizer.json` | Tokenizer model | ✅ Yes (already there) |
| `vocab.json` | Vocabulary mapping | ✅ Yes (already there) |
| `special_tokens_map.json` | Special tokens | ✅ Yes (already there) |
| `merges.txt` | BPE merges | ✅ Yes (already there) |
| `optimizer.pt` | Training optimizer state | ❌ No (optional) |
| `scheduler.pt` | Learning rate scheduler | ❌ No (optional) |
| `trainer_state.json` | Training metadata | ❌ No (optional) |
| Other `.pt` files | Training checkpoints | ❌ No (optional) |

## Troubleshooting

**If the backend still says "Loaded regression model" instead of "Loaded 2-bin classifier":**
- Make sure `model.safetensors` is in `./backend/models/reddit_year_model/`
- Check that `config.json` has `"num_labels": 2` (for 2-bin classification)
- Restart the backend after copying files

**If you get an error about missing tokenizer files:**
- Make sure all the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`, `merges.txt`) are present

