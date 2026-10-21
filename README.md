# GOP-4 Prediction Time Calculation

Cloud CPU measurement of ConvLSTM-UNet GOP-4 inference time.

## Run on GitHub Actions
1. Open **Actions** → **Measure Prediction Time (GOP-4)**
2. Click **Run workflow**
3. Download artifacts `gop4-prediction-times-ClassA` … `ClassD`
4. Put CSVs into local `Archive/Class Prediction Time/GOP-4/`
5. Run locally: `python delta_t_calculate_gop4.py`

## Data
- VVC: `Archive/VVC Reconstructed/All same resolution - GOP 4/`
- Model: `best_model_gop4.pt`
