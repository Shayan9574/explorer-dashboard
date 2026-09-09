# Ford Explorer Yield Management Dashboard

Interactive Streamlit dashboard for the Yield Management interview business case: price
elasticity estimation, evaluation of the two 2025 sales programs, competitive segmentation,
and 2026 recommendations. All figures are computed live from the source workbook.

## Deploy in 10 minutes

1. **Google Drive**
   - Upload `Data_Scientist_Interview_Business_Case.xlsx` to your Drive folder.
   - Right click the file, Share, set **Anyone with the link: Viewer**.
   - Copy the link. The FILE_ID is the long string between `/d/` and `/view`
     (e.g. `https://drive.google.com/file/d/FILE_ID/view`).

2. **GitHub**
   - Create a public repository (e.g. `explorer-dashboard`) and push all files in
     this folder, including `data/fallback.xlsx`.

3. **Streamlit Community Cloud** (free, public)
   - Go to https://share.streamlit.io, sign in with GitHub, New app.
   - Pick the repo, branch `main`, main file `app.py`, Deploy.
   - In App settings, Secrets, add:
     ```toml
     GDRIVE_FILE_ID = "paste_your_file_id_here"
     # optional, enables live AI analysis buttons:
     # ANTHROPIC_API_KEY = "sk-ant-..."
     ```
   - The app is now public at `https://<something>.streamlit.app`.

If the Drive download ever fails, the app automatically falls back to the copy in
`data/fallback.xlsx` and labels the source accordingly, so a live demo cannot break.

## Structure
- `app.py` — UI, eight tabs, all charts
- `data_loader.py` — Drive/fallback loading, monthly parsing, quarterly aggregation
- `analysis.py` — six elasticity models, arc elasticities, program decomposition
- `ai.py` — curated insights + optional live Claude analysis
- `data/fallback.xlsx` — bundled copy of the source workbook

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
