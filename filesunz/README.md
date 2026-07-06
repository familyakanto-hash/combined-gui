# RHA-Blended Concrete ML Toolkit — Combined App

A single Streamlit multipage app for the Department of Civil Engineering, MIST
thesis on machine-learning analysis of RHA-blended concrete. Combines all
five previously separate apps into one site with a proper landing page.

## Structure

```
gui/                                    (repo root)
├── Home.py                             ← landing page: logo, welcome, 2 category cards
├── utils.py                            ← shared theme/CSS + header + back-button helpers
├── mist_logo.jpg                       ← MIST crest, shown on the home page
├── pages/
│   ├── 1_Compressive_Strength.py
│   ├── 2_Flexural_Strength.py
│   ├── 3_Split_Tensile_Strength.py
│   ├── 4_Water_Absorption.py
│   └── 5_RCPT.py
├── dataset_compressive_strength.csv
├── dataset_flexural_strength.csv
├── dataset_split_tensile_strength_fixed.csv
├── dataset_water_absorption_28_fixed.csv
├── dataset_rcpt_28.csv
└── requirements.txt
```

This is **Streamlit's native multipage app convention**: any `.py` file inside
`pages/` automatically becomes a page, listed in the sidebar in filename
order. `Home.py` (or whichever file you run) is the entry point.

## How the navigation works

1. **Home.py** shows the MIST logo, a short blurb about the Department of
   Civil Engineering and the RHA thesis project, then two category cards:
   **Mechanical Properties** and **Durability Properties**.
2. Clicking a category reveals a small sub-menu (3 buttons for mechanical,
   2 for durability) right on the home page.
3. Clicking a specific property jumps straight to that page
   (`st.switch_page(...)`), which runs its own self-contained predictor +
   mix-designer tool — same logic as before, just re-skinned with the shared
   theme and a "← Back to Home" link.
4. The sidebar (Streamlit's default multipage nav) is always available too,
   so users can jump between pages directly without going back to Home.

## Deploying — replace your existing repo contents with this structure

1. In your `familyakanto/gui` GitHub repo, **delete** the old flat files
   (`rha_app.py`, `flexural_app.py`, `rcpt_app.py`, `split_tensile_app.py`,
   `water_absorption_app.py`) — their logic now lives inside `pages/`.
2. Upload this entire structure (`Home.py`, `utils.py`, `mist_logo.jpg`,
   the `pages/` folder with its 5 files, all 5 dataset CSVs, and
   `requirements.txt`) to the repo root, preserving the `pages/` folder.
3. In Streamlit Community Cloud, update the app's **main file path** to
   `Home.py` (Settings → General → Main file path).
4. Reboot the app.

## Notes
- Each page trains its own model on first load and caches it
  (`st.cache_resource`) — first visit to each page takes a few seconds,
  subsequent visits are instant for that session.
- All five pages keep the exact modelling pipeline and Optuna-tuned
  hyperparameters validated earlier — only the file paths, header, and
  styling changed to fit the combined app.
- Colors are muted versions of the MIST crest palette (navy, green, gold)
  for a professional, sober look, with subtle fade-in and hover-transition
  CSS for a more polished feel.
