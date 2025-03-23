# Document Table Extractor

A Streamlit-based application for extracting tables and structured content from documents (PDF, DOCX, or Images) using multiple OCR engines.

## Features

- Support for PDF, DOCX, and image files (PNG, JPG, JPEG, TIFF)
- Dual OCR processing using RapidOCR and EasyOCR
- Table extraction and cleaning
- Biomarker data extraction
- Export to CSV and Excel formats
- Document preview functionality
- Interactive web interface

## Docker Setup

### Prerequisites

- Docker installed on your system

### Building and Running with Docker

1. Build the Docker image:
```bash
docker compose up --build
```
1.5 If things take long with requirements
```
If dockling takes a lot of time with requirements file (figuring out dependendencies), remove from requirements and install it with "pip install dockling" within the container.
```
2. Run the container:
```bash
docker run -p 8501:8501 table-extractor
```

The application will be available at `http://localhost:8501`

## Local Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run table_extract.py
```

## Usage

1. Access the application through your web browser
2. Upload a document using the file uploader
3. View the document preview
4. Check the extracted tables and biomarkers in both RapidOCR and EasyOCR tabs
5. Download results in CSV or Excel format

## Notes

- The application creates a temporary `scratch` directory for processing files
- Both RapidOCR and EasyOCR results are provided for comparison
- Tables are presented in both original and cleaned formats
- Biomarker data is automatically extracted when possible

## Troubleshooting

- If you encounter memory issues, try adjusting Docker container resources
- For PDF processing issues, ensure the file is not password-protected
- Check the logs for detailed error messages



