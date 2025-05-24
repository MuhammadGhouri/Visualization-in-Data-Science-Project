
# Football Data Visualization Project

A comprehensive project for exploratory data analysis and visualization of European football data. The project covers player and team performance, attribute analysis, home advantage metrics, tactical comparisons, and more. All visualizations are automatically saved in the `visualizations` directory.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [How to Run](#how-to-run)
- [Reproducing the Visualizations](#reproducing-the-visualizations)
- [Output Files](#output-files)
- [Notes and Troubleshooting](#notes-and-troubleshooting)

---

## Project Structure

```
.
├── Football_Analysis_Group_15.py
├── visualizations/
│   └── [generated PNG and CSV files]
```

## Requirements

This project requires Python 3.7+ and the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

You can install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn
```

## Dataset Setup

You **must** have the following CSV files (from the European Soccer Database/Kaggle) available:

- `Player.csv`
- `Player_Attributes.csv`
- `Team.csv`
- `Team_Attributes.csv`
- `Match.csv`
- `League.csv`
- `Country.csv`

**Directory Structure:**
- By default, the script expects all these CSV files to be in `/home/ubuntu/football_data/`.

If your data is in a different location, either move your CSVs or edit the path strings in `Football_Analysis_Group_15.py` accordingly.

## How to Run

1. **Ensure all CSV files are in the expected directory.**
2. Open a terminal and navigate to the project folder.
3. Run the main analysis script:

```bash
python Football_Analysis_Group_15.py
```

The script will perform the analysis, print progress, and save all generated visualizations automatically.

## Reproducing the Visualizations

All visualizations are generated and saved as PNG files in the `visualizations/` folder.  
Each figure is named according to its content (e.g., `player_rating_distribution.png`, `team_tactics_radar.png`, etc.).

No manual intervention is needed; simply running the script will create the full set of outputs, including:
- Distribution histograms
- Heatmaps
- Radar charts (for both players and teams)
- Box plots
- Correlation matrices
- Line charts (trend analysis)
- Comprehensive dashboard

**To view a visualization:**  
Open any PNG file from the `visualizations/` directory using your image viewer.

**To reproduce everything:**  
Delete the contents of the `visualizations/` folder (if it exists), then re-run the script.

## Output Files

- **All generated images** are saved in `visualizations/` (created automatically).
- A sample match data CSV is saved as `visualizations/sample_match_data.csv`.
- Console output summarizes the analysis progress and results.

## Notes and Troubleshooting

- **Data Paths:**  
If you get a `FileNotFoundError`, check that your CSV files are in the correct directory. You can change the file paths in the script if needed.
- **Visualization Folder:**  
The script automatically creates the `visualizations/` directory if it does not exist.
- **Customization:**  
To use interactive visualizations, further development (using libraries like Plotly or Dash) is possible but not included in this version.
- **Limitations:**  
Some factors (e.g., stadium capacity, referee, travel distance) are not in the dataset and are thus not analyzed.

---

**Project by Group 15**  
Feedback welcome!  
If you have any issues, please open an issue in this repository or contact the author.

---
