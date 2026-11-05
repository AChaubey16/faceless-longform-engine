# Faceless Long-Form Video Engine (Groq + SDXL-Lightning + Kokoro-82M)

An automated, checkpointed pipeline for generating long-form YouTube videos across 3 distinct channel styles on free cloud GPUs (Kaggle Dual T4 or Google Colab T4):

1. **`ink_explainer_12m`** — 12-minute prehistoric & anthropology explainers (2D stick-figure caveman scenes + crisp white Pillow title/comparison cards).
2. **`history_in_minutes`** — 20-minute geopolitical & archival history documentaries (cinematic historical illustrations + warm archive date cards).
3. **`mr_nightmare_1hr`** — 60-minute psychological horror compilations (10 standalone stories, 35mm dark liminal photography + pitch-black story cards).

---

## Quick Start on Google Colab / Kaggle

### 1. Clone & Install Dependencies
```bash
!git clone https://github.com/YOUR_USERNAME/faceless-longform-engine.git
%cd faceless-longform-engine
!bash setup_cloud.sh
```

### 2. Set Your Free Groq API Key
Get a free API key at [console.groq.com](https://console.groq.com) (`llama-3.3-70b-versatile`):
```python
import os
os.environ["GROQ_API_KEY"] = "gsk_your_actual_groq_key_here"
```

### 3. Generate Videos
```bash
# 1. Ink Explainer Style (12 Mins)
!python main.py --topic "What Did Ancient Humans Do When It Rained All Week?" --preset ink_explainer_12m

# 2. History in Minutes Style (20 Mins)
!python main.py --topic "The ENTIRE History of MI6 in 15 Minutes" --preset history_in_minutes

# 3. Mr. Nightmare Style (1 Hour Compilation)
!python main.py --topic "10 True Night Shift Gas Station Horror Stories" --preset mr_nightmare_1hr
```
