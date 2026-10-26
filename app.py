import os
import io
import json
import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from ml_engine import SlothMLEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlothAPI")

app = FastAPI(title="SlothML - Tabular Foundation AI", version="1.0.0")

# Global engine instance and state cache
engine = SlothMLEngine()
current_data: Optional[pd.DataFrame] = None
current_filename: str = "demo_data.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def sanitize_df_preview(df: pd.DataFrame):
    """Safely extracts preview records without NaN or inf issues that crash JSON serialization."""
    head_df = df.head(5).copy()
    records = []
    for row in head_df.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v) or v is None:
                clean_row[str(k)] = "-"
            elif isinstance(v, (float, np.floating)):
                clean_row[str(k)] = "-" if (np.isnan(v) or np.isinf(v)) else round(float(v), 3)
            elif isinstance(v, (int, np.integer)):
                clean_row[str(k)] = int(v)
            else:
                clean_row[str(k)] = str(v)
        records.append(clean_row)
    return records


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>SlothML Dashboard Loading...</h2>")


@app.get("/api/demos")
async def get_demos():
    """Returns available preloaded demo datasets."""
    demos = [
        {
            "id": "saas_churn",
            "name": "SaaS Customer Churn",
            "category": "B2B SaaS / Business",
            "task": "Classification",
            "target": "Churn",
            "description": "Predict subscription cancellations based on usage, tenure, and support contacts.",
            "badge": "Popular"
        },
        {
            "id": "icu_readmission",
            "name": "ICU Patient Health Risk",
            "category": "Healthcare & Social Good",
            "task": "Classification",
            "target": "PatientOutcome",
            "description": "Classify 30-day critical care readmission risk using vital signs & history.",
            "badge": "High Impact"
        },
        {
            "id": "house_prices",
            "name": "Metro Real Estate Pricing",
            "category": "Real Estate / Finance",
            "task": "Regression",
            "target": "SalePrice",
            "description": "Estimate market listing valuation using square footage, location & quality scores.",
            "badge": "Regression"
        }
    ]
    return JSONResponse(content={"demos": demos})


@app.post("/api/load-demo")
async def load_demo(demo_id: str = Form(...)):
    """Loads one of the pre-loaded demo CSV files."""
    global current_data, current_filename
    file_map = {
        "saas_churn": "saas_churn.csv",
        "icu_readmission": "icu_readmission.csv",
        "house_prices": "house_prices.csv"
    }
    
    if demo_id not in file_map:
        raise HTTPException(status_code=400, detail="Unknown demo dataset requested.")

    file_path = os.path.join(DATASETS_DIR, file_map[demo_id])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Demo file not found.")

    df = pd.read_csv(file_path)
    current_data = df
    current_filename = file_map[demo_id]

    target_col, task_type = engine.detect_target_and_task(df)
    
    preview_rows = sanitize_df_preview(df)
    columns_info = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]

    return {
        "success": True,
        "filename": current_filename,
        "rows_count": len(df),
        "cols_count": len(df.columns),
        "detected_target": target_col,
        "detected_task": task_type,
        "columns": columns_info,
        "preview": preview_rows
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles custom CSV or Excel file uploads with auto-encoding and delimiter detection."""
    global current_data, current_filename
    try:
        contents = await file.read()
        filename = file.filename or "uploaded_data.csv"
        df = None

        if filename.lower().endswith((".csv", ".txt")):
            # Try multiple encodings and auto-detect separators (comma, semicolon, tab)
            for enc in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=enc, sep=None, engine='python')
                    if df is not None and len(df.columns) > 1:
                        break
                except Exception:
                    pass

            if df is None:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding_errors='replace')
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

        elif filename.lower().endswith((".xls", ".xlsx")):
            try:
                df = pd.read_excel(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not parse Excel spreadsheet: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Please upload CSV or Excel files.")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="The uploaded spreadsheet contains no data rows.")

        # Clean column names (strip whitespace, ensure no unnamed/empty headers, resolve duplicate names)
        cleaned_cols = []
        seen = {}
        for i, c in enumerate(df.columns):
            c_str = str(c).strip()
            if not c_str or c_str.lower().startswith("unnamed"):
                base = f"Feature_{i+1}"
            else:
                base = c_str

            if base in seen:
                seen[base] += 1
                cleaned_cols.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 1
                cleaned_cols.append(base)
        df.columns = cleaned_cols

        current_data = df
        current_filename = filename

        target_col, task_type = engine.detect_target_and_task(df)
        preview_rows = sanitize_df_preview(df)
        columns_info = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]

        return {
            "success": True,
            "filename": current_filename,
            "rows_count": len(df),
            "cols_count": len(df.columns),
            "detected_target": target_col,
            "detected_task": task_type,
            "columns": columns_info,
            "preview": preview_rows
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        # ✅ Close the string properly
        raise HTTPException(
            status_code=400, detail=f"Upload failed: {str(e)}"
        )


class BenchmarkRequest(BaseModel):
    target_col: Optional[str] = None


@app.post("/api/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    """Executes TabPFN vs Baseline benchmark, generates executive brief, and prepares simulator."""
    global current_data, engine
    if current_data is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a file or pick a demo first.")

    try:
        results = engine.fit_and_benchmark(current_data, target_col=req.target_col)
        return {
            "success": True,
            "results": results
        }
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=f"Modeling error: {str(e)}")


class SimulationRequest(BaseModel):
    features: Dict[str, Any]


@app.post("/api/simulate")
async def simulate(req: SimulationRequest):
    """Real-time scenario simulation triggered by frontend sliders."""
    global engine
    try:
        res = engine.simulate_what_if(req.features)
        return {
            "success": True,
            "simulation": res
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/api/export-code")
async def export_code():
    """Generates standalone Python code."""
    global engine
    code = engine.generate_cheat_code()
    return PlainTextResponse(content=code, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
