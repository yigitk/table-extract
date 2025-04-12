import logging
import pandas as pd
import re
from pathlib import Path
import io
from PIL import Image
import fitz  # PyMuPDF
import streamlit as st
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    EasyOcrOptions,
    AcceleratorOptions,  # Import AcceleratorOptions
    AcceleratorDevice    # Import AcceleratorDevice
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import settings # Import settings

_log = logging.getLogger(__name__)


# Document Processing Functions
def process_document(uploaded_file):
    """Process document with both OCR engines"""
    # Create temp directory for processing
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    easy_conv_res = None
    rapid_conv_res = None
    rapid_timings = None
    #easy_timings = None
    # Save uploaded file temporarily
    temp_path = output_dir / uploaded_file.name
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    rapid_pipeline_options = _configure_rapid_ocr()
    #easy_pipeline_options = _configure_easy_ocr()
    
    doc_converter_rapid = _create_document_converter(rapid_pipeline_options)
    #doc_converter_easy = _create_document_converter(easy_pipeline_options)

    # Enable timing
    settings.debug.profile_pipeline_timings = True

    # Process document with RapidOCR
    _log.info(f"Processing document with RapidOCR: {temp_path}")
    rapid_conv_res = doc_converter_rapid.convert(temp_path)
    rapid_timings = rapid_conv_res.timings
    
    # # Process document with EasyOCR
    # _log.info(f"Processing document with EasyOCR: {temp_path}")
    # easy_conv_res = doc_converter_easy.convert(temp_path)
    # easy_timings = easy_conv_res.timings
    
    doc_filename = rapid_conv_res.input.file.stem
    
    # Clean up temp file
    temp_path.unlink()
    
    return rapid_conv_res, easy_conv_res, doc_filename, rapid_timings #, easy_timings

def _configure_rapid_ocr():
    """Configure RapidOCR pipeline options"""
    #Configure accelerator
    accelerator_options = AcceleratorOptions(
        num_threads=4, # Experiment with different values (e.g., 4, 6, physical cores)
        device=AcceleratorDevice.CPU # Or GPU if available
    )
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options # Assign accelerator options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 2.0
    pipeline_options.ocr_options = RapidOcrOptions(
        force_full_page_ocr=False,
        use_cls=False,
        text_score=0.7
    )
    return pipeline_options

def _configure_easy_ocr():
    """Configure EasyOCR pipeline options"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 4.0
    pipeline_options.ocr_options = EasyOcrOptions(
        force_full_page_ocr=True,
        confidence_threshold=0.2
    )
    return pipeline_options

def _create_document_converter(pipeline_options):
    """Create document converter with given pipeline options"""
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

# File Preview Functions
@st.cache_data(show_spinner=False)
def preview_pdf_fitz(pdf_bytes: bytes, page: int = 1) -> tuple[Image.Image, str]:
    """PDF preview function using PyMuPDF"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_obj = pdf_document[page - 1]
        pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img, None
    except Exception as e:
        return None, f"Error previewing PDF: {str(e)}"
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def show_file_preview(uploaded_file):
    """Display a preview of the uploaded file"""
    file_type = uploaded_file.type
    
    if "image" in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    elif file_type == "application/pdf":
        try:
            pdf_bytes = uploaded_file.getvalue()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            pdf_document.close()
            
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            preview_image, error = preview_pdf_fitz(pdf_bytes, page)
            
            if error:
                st.error(error)
            elif preview_image:
                st.image(preview_image, caption=f"Page {page} of {total_pages}", use_column_width=True)
                
        except Exception as e:
            st.error(f"Could not generate PDF preview: {str(e)}")
    
    elif "word" in file_type:
        st.info("Preview not available for Word documents, but processing will continue.")

# Table Processing Functions
def clean_column_name(col_name: str) -> str:
    """Clean individual column name"""
    col = str(col_name)
    col = col.strip('|').strip(' ')
    col = ' '.join(col.split())
    return col

def rename_duplicate_columns(cols):
    """Rename duplicate columns by appending a suffix"""
    seen = {}
    new_cols = []
    for col in cols:
        cleaned = clean_column_name(col)
        if cleaned in seen:
            seen[cleaned] += 1
            new_cols.append(f"{cleaned}_{seen[cleaned]}")
        else:
            seen[cleaned] = 0
            new_cols.append(cleaned)
    return new_cols

def extract_biomarkers_from_text(text_content: str) -> pd.DataFrame:
    """Extract biomarker information from text content"""
    # Clean the text first
    text_content = clean_table_text(text_content)
    
    biomarkers = []
    
    # Split into lines
    lines = text_content.split('\n')
    
    # Skip header line if it exists
    start_idx = 1 if any(header in lines[0].upper() for header in ['TEST', 'RESULT', 'UNIT']) else 0
    
    # Compile unit pattern
    unit_pattern = r'(?i)(ng/L|mg/g|ug/dL|mg/dL|U/L|mg/L|ug/L|ng/mL|mm\[HG\]|IU/L|g/cm2|KG|%|Ω|L|ug/mL|' + \
                  r'g/dL|pg/mL|1000\s*cells/uL|fL|mosm/KG|10\*3/uL|x10\^12/L|uIU/L|ng/dL|mL|ug/g\[Hb\]|' + \
                  r'mm/h|cm|kPa|uIU/mL|IU/mL|mIU/mL|Type|Present|10\*6/mL|Score|mL/min/1\.73\s*m²|' + \
                  r'g/g\{creat\}|ratio|mg/mL|ph|clarity|color|kcal/kg/h|nm|pattern|count|s|arb\'U/mL|' + \
                  r'nmol/L|mg/mmol|umol/L|mmol/L|nmol/mL|pmol/L)'
    
    for line in lines[start_idx:]:
        # Split the line by | if it contains |, otherwise by spaces
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if not parts:
            continue
            
        test_name = parts[0] if parts else None
        result = None
        unit = None
        flag = None
        
        # Look for numeric result and unit in the parts
        for part in parts[1:]:
            # Skip empty parts or headers
            if not part or part.upper() in ['RESULT', 'UNITS', 'FLAG', 'REFERENCE INTERVAL']:
                continue
            
            # Look for unit first using the comprehensive pattern
            unit_match = re.search(unit_pattern, part)
            if unit_match:
                unit = unit_match.group(1)
                # Extract result from the remaining part
                result_part = part[:unit_match.start()].strip()
                number_match = re.search(r'([-+]?\d*\.?\d+)', result_part)
                if number_match:
                    result = number_match.group(1)
                continue
                
            # If no unit found, look for numeric value
            number_match = re.search(r'([-+]?\d*\.?\d+)', part)
            if number_match and not result:
                result = number_match.group(1)
                # Check remaining part for unit
                remaining = part[number_match.end():].strip()
                if remaining:
                    unit_match = re.search(unit_pattern, remaining)
                    if unit_match:
                        unit = unit_match.group(1)
                    elif remaining.upper() in ['HIGH', 'LOW', 'NORMAL']:
                        flag = remaining
                continue
            
            # If part is not numeric and result exists, it might be a flag
            if result and not flag and part.upper() in ['HIGH', 'LOW', 'NORMAL']:
                flag = part
        
        if test_name and result:
            unit_type = get_unit_type(unit) if unit else 'None'
            biomarkers.append({
                'Test': test_name,
                'Result': result,
                'Units': unit or '',
                'Unit Type': unit_type,
                'Flag': flag or ''
            })
    
    # Create DataFrame from extracted data
    if biomarkers:
        df = pd.DataFrame(biomarkers)
        # Clean up test names
        df['Test'] = df['Test'].str.strip(':').str.strip()
        return df
    return None
def clean_column_name(col_name: str) -> str:
    """Clean individual column name (without handling duplicates)."""
    col = str(col_name)
    # Remove leading/trailing spaces and special characters
    col = col.strip('|').strip(' ')
    # Replace multiple spaces with a single space
    col = ' '.join(col.split())
    return col

def rename_duplicate_columns(cols):
    """
    Rename duplicate columns by appending a suffix.
    For example, if "Current Result and Flag" appears twice,
    they become "Current Result and Flag", "Current Result and Flag_1".
    """
    seen = {}
    new_cols = []
    for col in cols:
        cleaned = clean_column_name(col)
        if cleaned in seen:
            seen[cleaned] += 1
            new_cols.append(f"{cleaned}_{seen[cleaned]}")
        else:
            seen[cleaned] = 0
            new_cols.append(cleaned)
    return new_cols

def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge columns that originally had the same header by concatenating their values row-wise.
    This function assumes the original header names (before renaming duplicates) are stored.
    """
    # First, clean names without mangling duplicates
    original_names = [clean_column_name(col) for col in df.columns]
    
    # Create a new DataFrame to store merged results
    new_df = pd.DataFrame()
    # Use order of first appearance in original_names
    unique_cols = list(dict.fromkeys(original_names))
    
    for col_name in unique_cols:
        # Get all column indices for this original column name
        indices = [i for i, name in enumerate(original_names) if name == col_name]
        if len(indices) == 1:
            # Only one column; copy it directly
            new_df[col_name] = df.iloc[:, indices[0]]
        else:
            # Multiple columns: merge by concatenating non-empty values row by row
            combined_series = df.iloc[:, indices].apply(
                lambda row: ' '.join(
                    str(x) for x in row if pd.notna(x) and str(x).strip() != ''
                ),
                axis=1
            )
            new_df[col_name] = combined_series.str.strip()
    
    return new_df

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to known formats after merging."""
    name_mappings = {
        'TEST': 'TESTS',
        'TEST NAME': 'TESTS',
        'EXAMINATION': 'TESTS',
        'VALUE': 'RESULT',
        'RESULTS': 'RESULT',
        'REFERENCE': 'REFERENCE INTERVAL',
        'REF INTERVAL': 'REFERENCE INTERVAL',
        'REF RANGE': 'REFERENCE INTERVAL',
        'REFERENCE RANGE': 'REFERENCE INTERVAL',
        'UNIT': 'UNITS',
        'FLAGS': 'FLAG'
    }
    
    # Clean names (these should now be unique if coming from merge_duplicate_columns)
    df.columns = [clean_column_name(col) for col in df.columns]
    # Convert to uppercase and apply mappings
    df.columns = [col.upper() for col in df.columns]
    df.columns = [name_mappings.get(col, col) for col in df.columns]
    return df

def clean_table_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and standardize table data.
    Returns a tuple: (original DataFrame with unique column names, cleaned DataFrame).
    """
    # Create a version of the original DataFrame with duplicate columns renamed for display
    original_df = df.copy()
    original_df.columns = rename_duplicate_columns(df.columns)
    
    try:
        # For merging, work with the original header names (without the _# suffix)
        df_merged = merge_duplicate_columns(df)
        # Standardize column names (which are now unique thanks to the merge)
        cleaned_df = standardize_column_names(df_merged)
        
        # Optionally, order columns as desired
        standard_columns = ['TESTS', 'RESULT', 'FLAG', 'UNITS', 'REFERENCE INTERVAL', 'LAB']
        existing_columns = [col for col in standard_columns if col in cleaned_df.columns]
        other_columns = [col for col in cleaned_df.columns if col not in standard_columns]
        cleaned_df = cleaned_df[existing_columns + other_columns]
    
    except Exception as e:
        st.error(f"Error during table cleaning: {str(e)}")
        return original_df, original_df
    
    return original_df, cleaned_df

def merge_reference_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Merge reference interval values from adjacent columns"""
    if 'REFERENCE INTERVAL' in df.columns:
        next_col = df.columns.get_loc('REFERENCE INTERVAL') + 1
        if next_col < len(df.columns):
            next_col_name = df.columns[next_col]
            if next_col_name not in ['UNITS', 'FLAG', 'LAB']:
                # Combine reference interval with next column if it contains numeric values
                next_col_data = df[next_col_name].astype(str)
                if next_col_data.str.contains(r'[\d.-]').any():
                    df['REFERENCE INTERVAL'] = df['REFERENCE INTERVAL'].fillna('') + ' ' + next_col_data.fillna('')
                    df = df.drop(columns=[next_col_name])
    
    return df


def get_max_columns(data):
    """Get the maximum number of columns in the data"""
    return max(len(row) for row in data)

def pad_row(row, max_cols):
    """Pad a row with empty strings if it has fewer columns than max_cols"""
    return row + [''] * (max_cols - len(row))

def clean_column_names(headers):
    """Clean column names by removing markdown formatting and handling duplicates"""
    # Remove markdown table formatting characters and whitespace
    cleaned = [h.strip('| ').strip() for h in headers]
    
    # Remove empty column names
    cleaned = [h if h else f'Column_{i+1}' for i, h in enumerate(cleaned)]
    
    # Handle duplicates by adding numbers
    seen = {}
    result = []
    for h in cleaned:
        if h in seen:
            seen[h] += 1
            result.append(f'{h}_{seen[h]}')
        else:
            seen[h] = 0
            result.append(h)
    
    return result

def clean_table_text(text_content: str) -> str:
    """Clean table-like text content"""
    # Split into lines and clean
    lines = []
    for line in text_content.split('\n'):
        line = line.strip()
        if line:
            # Remove excessive separators
            line = re.sub(r'\|+', '|', line)  # Replace multiple | with single |
            line = re.sub(r'\s+\|\s+', '|', line)  # Clean spaces around |
            line = line.strip('| ')  # Remove leading/trailing | and spaces
            # Skip separator lines
            if not re.match(r'^[-|]+$', line):
                lines.append(line)
    return '\n'.join(lines)

def get_unit_type(unit: str) -> str:
    """Determine if a unit is Standard or SI"""
    STANDARD_UNITS = {
        'ng/L', 'mg/g', 'ug/dL', 'mg/dL', 'U/L', 'mg/L', 'ug/L', 'ng/mL',
        'mm[HG]', 'IU/L', 'g/cm2', 'KG', '%', 'Ω', 'L', 'ug/mL', 'g/dL',
        'pg/mL', '1000 cells/uL', 'fL', 'mosm/KG', '10*3/uL', 'x10^12/L',
        'uIU/L', 'ng/dL', 'mL', 'ug/g[Hb]', 'mm/h', 'cm', 'kPa', 'uIU/mL',
        'IU/mL', 'mIU/mL', 'Type', 'Present', '10*6/mL', 'Score',
        'mL/min/1.73 m²', 'g/g{creat}', 'ratio', 'mg/mL', 'ph', 'clarity',
        'color', 'kcal/kg/h', 'nm', 'pattern', 'count', 's', "arb'U/mL"
    }
    
    SI_UNITS = {
        'nmol/L', 'mg/mmol', 'umol/L', 'mmol/L', 'nmol/mL', 'pmol/L'
    }
    
    # Clean and standardize unit for comparison
    unit = unit.lower().strip()
    unit = re.sub(r'\s+', '', unit)  # Remove all whitespace
    
    # Check against standardized units
    if unit in {u.lower() for u in STANDARD_UNITS}:
        return 'Standard'
    elif unit in {u.lower() for u in SI_UNITS}:
        return 'SI'
    return 'Other'

def process_tables_and_biomarkers(conv_res, biomarker_df, doc_filename, ocr_type, text_content):
    """Helper function to process tables and biomarkers for each OCR engine"""
    # Handle tables
    tables_found = False
    for table_ix, table in enumerate(conv_res.document.tables):
        try:
            table_df = table.export_to_dataframe()
            
            # Clean and standardize the table data
            original_df, cleaned_df = clean_table_data(table_df)
            
            # Display original table
            st.subheader(f"Table {table_ix + 1} (Original)")
            st.markdown(f"```\n{original_df.to_markdown()}\n```")
            st.dataframe(original_df)
            
            # Display cleaned table
            st.subheader(f"Table {table_ix + 1} (Cleaned)")
            st.markdown(f"```\n{cleaned_df.to_markdown()}\n```")
            st.dataframe(cleaned_df)
            
            # Create download buttons for CSV and Excel
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    f"Download Original Table {table_ix + 1} as CSV",
                    original_df.to_csv(index=False),
                    f"{doc_filename}-{ocr_type}-table-{table_ix + 1}-original.csv",
                    "text/csv",
                    key=f"{ocr_type}_original_csv_{table_ix}"
                )
            with col2:
                st.download_button(
                    f"Download Cleaned Table {table_ix + 1} as CSV",
                    cleaned_df.to_csv(index=False),
                    f"{doc_filename}-{ocr_type}-table-{table_ix + 1}-cleaned.csv",
                    "text/csv",
                    key=f"{ocr_type}_cleaned_csv_{table_ix}"
                )
            
            # Add Excel downloads in an expander
            with st.expander("Download Excel Files"):
                col1, col2 = st.columns(2)
                with col1:
                    excel_buffer = io.BytesIO()
                    original_df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        f"Download Original Table {table_ix + 1} as Excel",
                        excel_buffer.getvalue(),
                        f"{doc_filename}-{ocr_type}-table-{table_ix + 1}-original.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"{ocr_type}_original_excel_{table_ix}"
                    )
                with col2:
                    excel_buffer = io.BytesIO()
                    cleaned_df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        f"Download Cleaned Table {table_ix + 1} as Excel",
                        excel_buffer.getvalue(),
                        f"{doc_filename}-{ocr_type}-table-{table_ix + 1}-cleaned.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"{ocr_type}_cleaned_excel_{table_ix}"
                    )
            
            tables_found = True
            
        except Exception as e:
            st.error(f"Error processing table {table_ix}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Display biomarkers section
    if biomarker_df is not None and not biomarker_df.empty:
        st.subheader("Extracted Biomarkers")
        
        # Display summary statistics
        st.write("📊 Summary:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Biomarkers", len(biomarker_df))
        with col2:
            st.metric("Unique Tests", biomarker_df['Test'].nunique())
        
        # Display the data in different formats
        tab1, tab2, tab3 = st.tabs(["Table View", "Markdown View", "Raw Data"])
        
        with tab1:
            st.dataframe(
                biomarker_df,
                column_config={
                    "Test": st.column_config.TextColumn("Biomarker Test"),
                    "Result": st.column_config.NumberColumn("Value", format="%.2f"),
                    "Units": st.column_config.TextColumn("Unit")
                },
                hide_index=True
            )
        
        with tab2:
            st.markdown("### Biomarker Results")
            for _, row in biomarker_df.iterrows():
                st.markdown(f"""
                **{row['Test']}**  
                Value: {row['Result']} {row['Units']}
                """)
        
        with tab3:
            st.markdown("```\n" + biomarker_df.to_markdown() + "\n```")
        
        # Download section in an expander
        with st.expander("Download Options"):
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download as CSV",
                    biomarker_df.to_csv(index=False),
                    f"{doc_filename}-{ocr_type}-biomarkers.csv",
                    "text/csv",
                    use_container_width=True,
                    key=f"{ocr_type}_biomarkers_csv"
                )
            with col2:
                excel_buffer = io.BytesIO()
                biomarker_df.to_excel(excel_buffer, index=False)
                st.download_button(
                    "📥 Download as Excel",
                    excel_buffer.getvalue(),
                    f"{doc_filename}-{ocr_type}-biomarkers.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"{ocr_type}_biomarkers_excel"
                )
        
        st.markdown("---")
    else:
        st.warning("No biomarkers could be extracted from the text")
        # Display raw text for verification
        with st.expander("Show Raw Text"):
            st.text(text_content)
