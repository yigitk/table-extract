import logging
import tempfile
import shutil
from datetime import date as date_type # Use alias
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import re # Import re for column guessing
from fastapi import FastAPI, UploadFile, File, HTTPException, status

from models import ProcessedBiomarker
# Import functions from utils
# Also import the unit_pattern for guessing
# ADD load_known_biomarkers and match_biomarker
from utils import (
    process_document_for_api, extract_biomarkers_from_text,
    clean_table_data, clean_table_text, unit_pattern,
    load_known_biomarkers, match_biomarker, KNOWN_BIOMARKERS_FILE, # Import matching utils and file path constant
    extract_lab_date # <<< Import the new date extraction function
)

# --- Logging Setup ---
# Set logging level to DEBUG to see detailed logs from utils
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_log = logging.getLogger(__name__)

# --- Load Known Biomarkers on Startup ---
_log.info("Attempting to load known biomarkers on API startup...")
# Use the default path defined in utils, handle potential errors during startup
KNOWN_BIOMARKERS = load_known_biomarkers(KNOWN_BIOMARKERS_FILE)
if not KNOWN_BIOMARKERS:
    _log.warning("Known biomarkers list is empty or failed to load. Matching will not be performed.")
else:
    _log.info(f"Successfully loaded {len(KNOWN_BIOMARKERS)} biomarkers for matching.")
# --- End Biomarker Loading ---


app = FastAPI(
    title="Document Processing API",
    description="Extracts tables and biomarkers from PDF, DOCX, and Image files, including biomarker matching.",
    version="0.1.2", # Incremented version
)

# --- Helper Function for Column Guessing ---
def find_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempts to identify Test, Result, and Unit columns in a DataFrame.

    Returns:
        A tuple (test_col_name, result_col_name, unit_col_name),
        where elements are None if the column couldn't be identified.
    """
    test_col, result_col, unit_col = None, None, None
    cols = df.columns.tolist()
    _log.debug(f"Attempting to find columns in: {cols}")

    # Prioritize standard names first
    if 'TESTS' in cols: test_col = 'TESTS'
    if 'RESULT' in cols: result_col = 'RESULT'
    if 'UNITS' in cols: unit_col = 'UNITS'

    # If standard names not fully found, try guessing
    # Guess Test Column (often first, contains text)
    if test_col is None and cols:
        potential_test_cols = [c for c in cols if any(kw in str(c).upper() for kw in ['TEST', 'ANALYTE', 'PARAM', 'NAME', 'ASSAY'])]
        if potential_test_cols:
            test_col = potential_test_cols[0] # Take first match
            _log.debug(f"Guessed Test column: {test_col}")
        elif isinstance(cols[0], str): # Fallback to first column if it's text-based
             # Basic check: does the first column contain mostly non-numeric data?
             try:
                 if pd.to_numeric(df[cols[0]], errors='coerce').isna().mean() > 0.6: # If >60% are non-numeric
                      test_col = cols[0]
                      _log.debug(f"Guessed Test column (fallback to first): {test_col}")
             except Exception: pass # Ignore errors during check

    # Guess Result Column (contains numbers or specific qualitative words)
    if result_col is None:
        potential_result_cols = [c for c in cols if any(kw in str(c).upper() for kw in ['RESULT', 'VALUE', 'MEASURE'])]
        if potential_result_cols:
            result_col = potential_result_cols[0]
            _log.debug(f"Guessed Result column: {result_col}")
        else: # Fallback: look for columns with high numeric content
             for col in cols:
                 if col == test_col: continue # Don't reuse test column
                 try:
                     # Check if a good portion of the column can be numeric
                     numeric_ratio = pd.to_numeric(df[col], errors='coerce').notna().mean()
                     if numeric_ratio > 0.5: # If >50% are numeric
                          result_col = col
                          _log.debug(f"Guessed Result column (fallback based on numeric content): {result_col}")
                          break
                 except Exception: continue

    # Guess Unit Column (matches unit pattern)
    if unit_col is None:
        potential_unit_cols = [c for c in cols if any(kw in str(c).upper() for kw in ['UNIT'])]
        if potential_unit_cols:
             unit_col = potential_unit_cols[0]
             _log.debug(f"Guessed Unit column: {unit_col}")
        else: # Fallback: Check column content against unit regex
            for col in cols:
                if col == test_col or col == result_col: continue # Don't reuse
                try:
                    # Check if a significant portion of non-null values match the unit pattern
                    matches = df[col].dropna().astype(str).apply(lambda x: bool(unit_pattern.fullmatch(x.strip())))
                    if matches.any() and matches.mean() > 0.3: # If at least one matches and >30% of non-null match
                        unit_col = col
                        _log.debug(f"Guessed Unit column (fallback based on content match): {unit_col}")
                        break
                except Exception: continue

    _log.debug(f"Found columns - Test: {test_col}, Result: {result_col}, Unit: {unit_col}")
    return test_col, result_col, unit_col
# --- End Helper Function ---

@app.post(
    "/process_document/",
    response_model=List[ProcessedBiomarker],
    summary="Process a document and extract biomarkers",
    tags=["Processing"],
)
async def process_document_endpoint(
    file: UploadFile = File(..., description="The document file (PDF, DOCX, PNG, JPG, TIFF) to process.")
) -> List[ProcessedBiomarker]:
    """
    Uploads a document, processes it using RapidOCR, extracts structured data
    primarily from tables, performs biomarker matching against a known list,
    extracts the lab date (reported/collected), and returns a list of
    structured biomarker data. Includes intelligent column guessing for tables.
    """
    allowed_content_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # DOCX
        "image/png", "image/jpeg", "image/tiff",
    ]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Please upload PDF, DOCX, PNG, JPG, or TIFF.",
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="docproc_api_"))
    temp_path = temp_dir / (file.filename or "uploaded_file") # Handle missing filename
    processing_date = date_type.today() # Get processing date

    try:
        # Save uploaded file
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        _log.info(f"Temporarily saved uploaded file to: {temp_path}")

        # --- Core Processing ---
        conv_res = await process_document_for_api(temp_path)

        if not conv_res or not hasattr(conv_res, 'document'):
             _log.error(f"Document processing failed or returned invalid result for {temp_path}")
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document processing failed.")

        # --- Extract Lab Date --- <<< ADDED
        lab_date: Optional[date_type] = None
        try:
            lab_date = extract_lab_date(conv_res) # Call the function from utils
            _log.info(f"Extracted lab date from document: {lab_date}")
        except Exception as date_exc:
            # Log error but continue processing biomarkers
            _log.exception(f"Error during lab date extraction: {date_exc}")
        # --- End Extract Lab Date ---

        processed_biomarkers: List[ProcessedBiomarker] = []

        # --- Process Structured Tables ---
        if hasattr(conv_res.document, 'tables') and conv_res.document.tables:
            _log.info(f"Found {len(conv_res.document.tables)} tables in the document.")
            for i, table in enumerate(conv_res.document.tables):
                _log.info(f"--- Processing Table {i+1} ---")
                try:
                    raw_table_df = table.export_to_dataframe()
                    if raw_table_df.empty:
                        _log.info(f"Table {i+1} is empty.")
                        continue

                    # Use the cleaning pipeline (which includes header promotion)
                    _original_df, cleaned_df = clean_table_data(raw_table_df)

                    if cleaned_df.empty:
                        _log.warning(f"Table {i+1} resulted in an empty DataFrame after cleaning.")
                        continue

                    # --- INTELLIGENT COLUMN IDENTIFICATION ---
                    test_col_name, result_col_name, unit_col_name = find_columns(cleaned_df)

                    if not test_col_name or not result_col_name:
                        _log.warning(f"Could not reliably identify Test and Result columns in Table {i+1}. Columns found: {cleaned_df.columns.tolist()}. Skipping biomarker mapping for this table.")
                        continue
                    # --- END INTELLIGENT COLUMN IDENTIFICATION ---

                    _log.info(f"Table {i+1}: Using Test='{test_col_name}', Result='{result_col_name}', Unit='{unit_col_name}'")

                    # Map rows using identified column names
                    for _, row in cleaned_df.iterrows():
                        # Use identified column names, checking for NaNs
                        test_name = str(row[test_col_name]) if pd.notna(row[test_col_name]) else None
                        result_raw = str(row[result_col_name]) if pd.notna(row[result_col_name]) else None

                        if not test_name or result_raw is None: # Skip rows without essential data
                            continue

                        # *** START BIOMARKER MATCHING ***
                        matched_id: Optional[str] = None
                        is_matched: bool = False
                        if KNOWN_BIOMARKERS: # Only match if biomarkers were loaded
                            matched_id, is_matched = match_biomarker(test_name, KNOWN_BIOMARKERS)
                        # *** END BIOMARKER MATCHING ***

                        # Convert result to numeric
                        value_numeric: Optional[float] = None
                        # Check if utils already created Result_Numeric (might happen in text fallback)
                        # Or attempt conversion from identified result column
                        if 'Result_Numeric' in cleaned_df.columns and pd.notna(row['Result_Numeric']):
                             value_numeric = float(row['Result_Numeric'])
                        else:
                             try:
                                 # Use regex for more robust cleaning before conversion
                                 cleaned_result = re.sub(r'[<>=≤≥,]', '', str(result_raw)).strip()
                                 value_numeric = float(cleaned_result)
                             except (ValueError, TypeError):
                                 _log.debug(f"Could not convert table result '{result_raw}' to float for test '{test_name}'. Setting value to None.")
                                 value_numeric = None

                        # Get Unit using identified unit column name (if found)
                        unit = str(row[unit_col_name]) if unit_col_name and unit_col_name in cleaned_df.columns and pd.notna(row[unit_col_name]) else None

                        # --- Create ProcessedBiomarker with Lab Date --- <<< MODIFIED
                        biomarker = ProcessedBiomarker(
                            rawBiomarkerName=test_name,
                            value=value_numeric,
                            unit=unit,
                            biomarkerId=matched_id,
                            matched=is_matched,
                            labDate=lab_date,           # Assign extracted lab date
                            processingDate=processing_date # Assign processing date
                        )
                        processed_biomarkers.append(biomarker)

                except Exception as table_exc:
                    _log.exception(f"Error processing table {i+1}: {table_exc}")

        # --- (Optional) Fallback: Process Raw Text ---
        if not processed_biomarkers:
            _log.warning("No biomarkers extracted from tables, attempting fallback using raw text export.")
            text_content = ""
            if hasattr(conv_res.document, 'export_to_text'):
                 try:
                     text_content = conv_res.document.export_to_text()
                 except Exception as text_exc:
                      _log.error(f"Failed to export document to text: {text_exc}")

            if text_content:
                cleaned_text = clean_table_text(text_content)
                _log.debug(f"Raw text content for fallback processing (cleaned):\n{cleaned_text[:500]}...")
                biomarker_df_text = extract_biomarkers_from_text(cleaned_text)

                if biomarker_df_text is not None and not biomarker_df_text.empty:
                    _log.info(f"Fallback: Mapping {len(biomarker_df_text)} biomarkers found via regex on raw text.")
                    for _, row in biomarker_df_text.iterrows():
                        test_name_text = str(row['Test']) if 'Test' in biomarker_df_text.columns and pd.notna(row['Test']) else None
                        if not test_name_text: continue # Need a test name

                        # *** START BIOMARKER MATCHING (Fallback) ***
                        matched_id_text: Optional[str] = None
                        is_matched_text: bool = False
                        if KNOWN_BIOMARKERS: # Only match if biomarkers were loaded
                            matched_id_text, is_matched_text = match_biomarker(test_name_text, KNOWN_BIOMARKERS)
                        # *** END BIOMARKER MATCHING (Fallback) ***

                        value_numeric_text = float(row['Result_Numeric']) if 'Result_Numeric' in biomarker_df_text.columns and pd.notna(row['Result_Numeric']) else None
                        if value_numeric_text is None:
                             result_raw_text = str(row['Result']) if 'Result' in biomarker_df_text.columns and pd.notna(row['Result']) else None
                             if result_raw_text:
                                 try:
                                     # Use regex for more robust cleaning before conversion
                                     cleaned_result = re.sub(r'[<>=≤≥,]', '', result_raw_text).strip()
                                     value_numeric_text = float(cleaned_result)
                                 except (ValueError, TypeError): value_numeric_text = None

                        # --- Create ProcessedBiomarker with Lab Date (Fallback) --- <<< MODIFIED
                        biomarker_text = ProcessedBiomarker(
                            rawBiomarkerName=test_name_text,
                            value=value_numeric_text,
                            unit=str(row['Units']) if 'Units' in biomarker_df_text.columns and pd.notna(row['Units']) else None,
                            biomarkerId=matched_id_text,
                            matched=is_matched_text,
                            labDate=lab_date,           # Assign extracted lab date
                            processingDate=processing_date # Assign processing date
                        )
                        processed_biomarkers.append(biomarker_text)
            else:
                 _log.warning("No text content available for fallback processing.")

        if not processed_biomarkers:
             _log.warning(f"No biomarkers could be extracted from document {file.filename} using table or text methods.")

        _log.info(f"Returning {len(processed_biomarkers)} processed biomarkers including lab date '{lab_date}'.")
        return processed_biomarkers
    except HTTPException:
         raise # Re-raise HTTP exceptions directly
    except Exception as e:
        _log.exception(f"Unhandled error processing file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during document processing: {str(e)}",
        )
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            _log.info(f"Cleaned up temporary directory: {temp_dir}")

# --- How to run ---
# uvicorn api:app --reload --host 0.0.0.0 --port 8000 