#!/usr/bin/env python3
"""FastAPI bridge that exposes alerts/status to the React frontend.
It uses Supabase as the datastore so the Python monitors (or test clients)
can insert alerts, and the frontend can fetch them via HTTP.
"""
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from dotenv import load_dotenv

# Reuse Supabase client helper
from supabase_client import supabase

load_dotenv()

app = FastAPI(title="Threat Detection API", version="1.0.0")

# Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AlertIn(BaseModel):
    timestamp: Optional[str] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    protocol: Optional[str] = None
    threat_level: int = Field(0, ge=0, le=5)
    threats: Optional[List[str]] = None
    packet_size: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/alerts")
def list_alerts(limit: int = 50):
    try:
        query = supabase.table("alerts").select("*").order("timestamp", desc=True).limit(limit)
        resp = query.execute()
        # resp.data is list of dicts
        return {"items": resp.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alerts")
def create_alert(alert: AlertIn):
    try:
        payload = alert.dict()
        if not payload.get("timestamp"):
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Normalize array for threats
        if payload.get("threats") is None and payload.get("extra") and isinstance(payload["extra"], dict):
            # try populate from extra
            ts = payload["extra"].get("threats")
            if isinstance(ts, list):
                payload["threats"] = ts
        resp = supabase.table("alerts").insert(payload).execute()
        return {"inserted": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
