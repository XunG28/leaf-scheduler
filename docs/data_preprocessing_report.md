The carbon intensity is calculated using a production-based approach, which considers only the electricity generated within the TransnetBW control area. This method is appropriate for real-time scheduling optimization, as it reflects the instantaneous carbon footprint of local power generation. Future work could extend this to a consumption-based approach by incorporating cross-border electricity flows.



# LEAF: Data Preprocessing Report

## 1. Overview

This document describes the data preprocessing pipeline for the LEAF (Lightweight Eco-Aware Framework) project, which aims to predict carbon intensity for energy-aware scheduling of compute jobs and laboratory activities.

**Project Context:**  
Application demo for the position "Wissenschaftlicher Mitarbeiter im Bereich Energieeffiziente KI-basierte Planung von Computejobs und Laboraktivitäten" (Kennzahl 6051) at Hochschule Karlsruhe (HKA).

---

## 2. Data Source

### 2.1 Primary Data Source

| Attribute | Value |
|-----------|-------|
| **Provider** | SMARD (Strommarktdaten) - Bundesnetzagentur |
| **URL** | https://www.smard.de |
| **Region** | Control Area: TransnetBW (Germany) |
| **Category** | Electricity Generation |
| **Resolution** | 15-minute intervals (Quarterhour) |
| **Time Range** | 2024-03-02 00:00 → 2026-03-08 23:45 |
| **Total Records** | ~70,000 observations |

### 2.2 Data Fields

| Column | Description | Unit |
|--------|-------------|------|
| Start date | Interval start timestamp | datetime |
| End date | Interval end timestamp | datetime |
| Biomass | Biomass power generation | MWh |
| Hydropower | Hydroelectric generation | MWh |
| Wind onshore | Onshore wind generation | MWh |
| Photovoltaics | Solar PV generation | MWh |
| Other renewable | Other renewable sources | MWh |
| Nuclear | Nuclear generation | MWh |
| Hard coal | Hard coal generation | MWh |
| Fossil gas | Natural gas generation | MWh |
| Hydro pumped storage | Pumped-storage hydroelectricity | MWh |
| Other conventional | Other conventional sources | MWh |

---

## 3. CO₂ Emission Factors

### 3.1 Source

The emission factors are based on the official data published by the German Federal Environment Agency:

> **Umweltbundesamt (2023).** *Entwicklung der spezifischen Treibhausgas-Emissionen des deutschen Strommix.*  
> Available at: https://www.umweltbundesamt.de/publikationen

### 3.2 Applied Emission Factors

| Energy Source | Emission Factor (g CO₂/kWh) | Classification |
|---------------|----------------------------|----------------|
| Hard coal (Steinkohle) | **820** | Fossil fuel |
| Fossil gas (Erdgas) | **420** | Fossil fuel |
| Other conventional | **700** | Fossil fuel (conservative estimate) |
| Biomass | 0 | Renewable |
| Hydropower | 0 | Renewable |
| Wind onshore | 0 | Renewable |
| Photovoltaics | 0 | Renewable |
| Other renewable | 0 | Renewable |
| Nuclear | 0 | Low-carbon (phased out in Germany) |
| Hydro pumped storage | 0 | Storage (no direct emissions) |

### 3.3 Justification

1. **Hard coal (820 g/kWh):** Direct value from UBA 2023 report for German hard coal power plants.

2. **Fossil gas (420 g/kWh):** Direct value from UBA 2023 report for German natural gas power plants.

3. **Other conventional (700 g/kWh):** Conservative estimate for mixed fossil fuels (oil, diesel backup generators). This value lies between natural gas (420) and hard coal (820) to account for uncertainty.

4. **Renewables (0 g/kWh):** Operational emissions are zero. Life-cycle emissions are excluded from this analysis, following the common practice in grid carbon intensity calculations.

5. **Nuclear (0 g/kWh):** Operational CO₂ emissions are negligible. Note: Germany completed its nuclear phase-out in April 2023, so this column contains no active generation in the dataset.

6. **Hydro pumped storage (0 g/kWh):** Storage systems do not directly emit CO₂. The electricity consumed for pumping is accounted for elsewhere in the grid.

---

## 4. Calculation Methodology

### 4.1 Total Generation

$$\text{Total Generation} = \sum_{i \in \text{all sources}} \text{Generation}_i$$

Includes all generation sources: fossil fuels, renewables, nuclear, and pumped storage.

### 4.2 Renewable Share

$$\text{Renewable Share (\%)} = \frac{\sum_{i \in \text{renewables}} \text{Generation}_i}{\text{Total Generation}} \times 100$$

Where renewables include: Biomass, Hydropower, Wind onshore, Photovoltaics, Other renewable.

### 4.3 CO₂ Intensity

$$\text{CO}_2\text{ Intensity (g/kWh)} = \frac{\sum_{i \in \text{fossil}} \text{Generation}_i \times \text{EF}_i}{\text{Total Generation}}$$

Where EF = Emission Factor for each fossil fuel source.

---

## 5. Data Preprocessing Steps

### Step 1: Data Loading
- Load CSV file with semicolon delimiter (`;`)
- Encoding: UTF-8

### Step 2: Timestamp Conversion
- Parse `Start date` and `End date` columns to datetime format
- Handle mixed date formats from SMARD export

### Step 3: Numeric Value Cleaning
- Convert missing values (`-`, empty strings) to `0.0`
- Handle German number format (`1.234,56` → `1234.56`) if present
- Convert all generation columns to float type

### Step 4: Calculate Derived Metrics
- Total Generation (MWh)
- Renewable Generation (MWh)
- CO₂ Emissions (weighted sum)
- Renewable Share (%)
- CO₂ Intensity (g/kWh)

### Step 5: Data Quality Validation
- Filter out rows with zero total generation (avoid division by zero)
- Log statistics for quality assurance

---

## 6. Dataset Split

For machine learning experiments, the data is split as follows:

| Dataset | Time Range | Duration | Purpose |
|---------|------------|----------|---------|
| **Training** | 2024-03-02 → 2026-01-31 | ~22 months | Model training |
| **Validation** | 2026-02-01 → 2026-03-01 | ~1 month | Hyperparameter tuning |
| **Test** | 2026-03-02 → 2026-03-08 | ~1 week | Final evaluation |

### Rationale:
- **Temporal split** preserves time series integrity (no data leakage)
- **Validation period** is recent enough to reflect current patterns
- **Test period** represents unseen future data for realistic evaluation

---

## 7. Output Files

| File | Location | Description |
|------|----------|-------------|
| `energy_data_full.csv` | `data/processed/` | Complete processed dataset |

### Output Columns:
- `Start date`: Interval start timestamp
- `End date`: Interval end timestamp
- `Total_Generation_MWh`: Total power generation
- `Renewable_Generation_MWh`: Renewable power generation
- `Renewable_Share_pct`: Renewable share (%)
- `CO2_Intensity_gkWh`: Carbon intensity (g CO₂/kWh)

---

## 8. Reproducibility

### Code Location
```
leaf-scheduler/
├── leaf/data/preprocessor.py    # Core preprocessing module
└── scripts/process_raw_data.py  # Execution script
```

### Execution Command
```bash
cd leaf-scheduler
python scripts/process_raw_data.py
```

### Dependencies
- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0

---

## 9. References

[1] Umweltbundesamt (2023). *Entwicklung der spezifischen Treibhausgas-Emissionen des deutschen Strommix in den Jahren 1990–2022.* Dessau-Roßlau: Umweltbundesamt.

[2] SMARD - Strommarktdaten. Bundesnetzagentur. https://www.smard.de

[3] TransnetBW GmbH. Control Area Data. https://www.transnetbw.de

---

## 10. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-12 | Yican Guo | Initial version |
