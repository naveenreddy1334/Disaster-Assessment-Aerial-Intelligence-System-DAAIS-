# Disaster Assessment Aerial Intelligence System (DAAIS)

A smart and efficient aerial intelligence system designed for rapid disaster assessment using drone imagery and computer vision. This project utilizes deep learning and geospatial analysis to detect and classify disaster-impacted areas, enabling timely responses and resource allocation.

## Project Overview

Disaster Assessment Aerial Intelligence System (DAAIS) aims to:
- Automate the detection of disaster zones from aerial/drone images.
- Classify the severity of damage using machine learning.
- Provide visual and geospatial insights for emergency response teams.
- Improve speed and accuracy of disaster assessments.
- Integrate thermal imaging analysis for heat signature detection and fire monitoring.
- Enable multi-spectral analysis for comprehensive damage assessment across different wavelengths.

## Features

- **Disaster Detection** – Identify damaged zones from aerial imagery.
- 🗺**Geospatial Mapping** – Visualize affected areas on interactive maps.
- **Dashboard Interface** – User-friendly interface to view results.
- **Modular Design** – Easily extendable and adaptable system.
- **Real-time Processing** – Process drone footage with minimal latency.
- **Multi-Disaster Support** – Handles fires, floods, building collapses, and more.
- **Cross-platform Access** – Web, mobile, and field deployment compatibility.
- **Automated Report Generation** – Generate comprehensive PDF and JSON reports for documentation and analysis.

## 🛠️ Technologies Used

- Python
- TensorFlow / PyTorch
- OpenCV
- Leaflet / OpenStreetMap (Geospatial Visualization)
- HTML/CSS/JavaScript (Frontend)
- NumPy / Pandas (Data Processing)
- **ReportLab / WeasyPrint** (PDF Report Generation)
- **JSON** (Structured Data Export)

## 🖼️ Sample Output

https://github.com/user-attachments/assets/63c3ac4e-f53c-490c-a9e7-b3bc8a06bd97

https://github.com/user-attachments/assets/1271205c-0fb9-43c7-9bc0-571b53ebd6fa

**Example outputs include:**
- Before vs After disaster detection comparisons
- Interactive map-based visualizations  
- Damage severity heat maps
- Automated Report generated

## 📁 Folder Structure

```bash
📦 DAAIS/
├── 📂 scripts/            # Automation and deployment scripts
│   ├── 📜 thermal.py      # Thermal imaging processing script
│   ├── 📜 disaster.py     # Main disaster detection script
│   ├── 📜 startup.py      # Application startup script
│   ├── 📜 train_model.py  # Model training automation
│   ├── 📜 generate_report.py  # PDF report generation script
│   └── 📜 export_data.py  # JSON data export utility
├── 📜 requirements.txt    # Python dependencies
├── 📜 start.service       # System service configuration
├── 📜 config.yaml        # Configuration settings
└── 📜 README.md          # Project documentation
```

## Getting Started

### Prerequisites

Before getting started with DAAIS, ensure your runtime environment meets the following requirements:

- Python 3.8+ (recommended: Python 3.9+)
- pip or conda package manager
- GDAL (for geospatial operations)

### Installation

Install DAAIS using the following steps:

**Build from source:**

1. Clone the DAAIS repository:
```bash
git clone https://github.com/naveenreddy1334/Disaster-Assessment-Aerial-Intelligence-System-DAAIS-
```

2. Navigate to the project directory:
```bash
cd Disaster-Assessment-Aerial-Intelligence-System-DAAIS-
```

3. Create a virtual environment:
```bash
python -m venv daais_env
source daais_env/bin/activate  # On Windows: daais_env\Scripts\activate
```

4. Install the project dependencies:
```bash
pip install -r requirements.txt
```

5. Install additional geospatial dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# On macOS
brew install gdal

# On Windows (using conda)
conda install -c conda-forge gdal
```

### Usage

**Run the Web Application:**
```bash
cd webapp
python app.py
```

**Process Aerial Imagery:**
```bash
python scripts/process_imagery.py --input_dir data/raw --output_dir data/processed
```

**Train a New Model:**
```bash
python scripts/train_model.py --config configs/disaster_detection.yaml
```

**Generate Assessment Reports:**
```bash
python scripts/generate_report.py --input_analysis results/analysis.json --format pdf --output reports/disaster_assessment.pdf
```

**Export Data in JSON Format:**
```bash
python scripts/export_data.py --analysis_id 12345 --format json --output data/exports/assessment_data.json
```

Open http://localhost:5000 in your browser to access the dashboard.

### Testing

Run the test suite using the following command:
```bash
python -m pytest tests/ -v
```

**Run specific test categories:**
```bash
# Test model inference
python -m pytest tests/test_models.py

# Test data preprocessing
python -m pytest tests/test_preprocessing.py

# Test web application
python -m pytest tests/test_webapp.py
```

## Technical Specifications

### Supported Disaster Types
- Wildfires and smoke detection
- Flood damage assessment
- Collapsed buildings and rubble
- Traffic accidents
- General infrastructure damage

### Input Formats
- High-resolution RGB imagery
- Thermal/Infrared imagery
- Multi-spectral satellite data
- Real-time drone video feeds

### Output Formats
- Damage severity maps
- Geospatial damage reports
- Statistical summaries
- Interactive web visualizations
- **PDF Reports** – Comprehensive assessment documents with charts, maps, and analysis
- **JSON Data Files** – Structured data exports for integration with other systems and APIs

### Performance Metrics
- **Detection Accuracy:** 95%+ for major damage categories
- **Processing Speed:** Real-time analysis capability
- **Coverage:** Up to 45 km² per analysis session
- **Resolution:** Support for sub-meter resolution imagery

## Contributing

We welcome contributions to improve DAAIS! Here's how you can help:

- **Join the Discussions:** Share your insights, provide feedback, or ask questions.
- **Report Issues:** Submit bugs found or log feature requests for the DAAIS project.
- **Submit Pull Requests:** Review open PRs, and submit your own PRs.

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/disaster-detection-improvement`)
3. Commit your changes (`git commit -am 'Add new disaster type detection'`)
4. Push to the branch (`git push origin feature/disaster-detection-improvement`)
5. Create a Pull Request
6. Ensure all tests pass
7. Update documentation as needed
8. Follow code style guidelines (PEP 8 for Python)
