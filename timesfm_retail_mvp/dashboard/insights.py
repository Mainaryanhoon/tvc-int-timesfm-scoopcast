from __future__ import annotations
import pandas as pd
import numpy as np


def generate_executive_insights(mode: str, result_dict: dict, cfg, series_names: list[str], horizon: int) -> dict:
    """Generates plain-English executive takeaways based on forecast results."""
    takeaways = []
    
    if mode == "Inventory":
        inv = result_dict
        fc = inv["forecast"]
        current_stock = inv["current_inventory"]
        total_demand = float(fc.point.sum())
        safety_buffer = float(inv["safety_stock"])
        shortage_deficit = max(total_demand - current_stock, 0.0)
        risk_days = int(np.sum(inv["projected_inventory"] < safety_buffer))
        
        takeaways.append(f"<b>30-Day Outlook:</b> Total expected customer demand is <b>{total_demand:,.0f} units</b> against starting stock of <b>{current_stock:,.0f} units</b>.")
        
        if shortage_deficit > 0 or risk_days > 0:
            takeaways.append(f"<b>Stock Alert:</b> Shelf stock falls below safety buffer threshold ({safety_buffer:,.0f} units) for <b>{risk_days} days</b>.")
            takeaways.append(f"<b>Supply Action:</b> Immediate purchase order recommended for at least <b>{shortage_deficit:,.0f} units</b>.")
        else:
            takeaways.append("<b>Stock Alert:</b> Current inventory levels are healthy across the full forecast horizon.")
            takeaways.append("<b>Supply Action:</b> Maintain standard replenishment cadence without expedited reorders.")
            
        return {
            "title": f"Inventory & Supply Chain Brief — {series_names[0]}",
            "takeaways": takeaways,
            "status": "SHORTAGE RISK" if shortage_deficit > 0 else "HEALTHY STOCK"
        }

    else:
        base_res = result_dict["base"] if isinstance(result_dict, dict) and "base" in result_dict else result_dict
        scen_res = result_dict["scenario"] if isinstance(result_dict, dict) and "scenario" in result_dict else result_dict
        has_what_if = result_dict.get("has_what_if", False) if isinstance(result_dict, dict) else False

        base_tot = float(base_res.point.sum())
        scen_tot = float(scen_res.point.sum())
        lift_pct = ((scen_tot - base_tot) / base_tot * 100.0) if base_tot > 0 else 0.0
        
        unit = cfg.unit
        takeaways.append(f"<b>Baseline Expectation:</b> Projected {cfg.name.lower()} total across selected series is <b>{scen_tot:,.0f} {unit}</b> over {horizon} days.")
        
        if has_what_if:
            takeaways.append(f"<b>Scenario Impact:</b> Simulated promotional campaign yields an estimated <b>+{lift_pct:.1f}% demand lift</b> over the baseline plan.")
        else:
            takeaways.append("<b>Scenario Impact:</b> Running under standard organic demand baseline without active promotional overrides.")
            
        p10_tot = float(scen_res.quantiles[:, :, 0].sum())
        p90_tot = float(scen_res.quantiles[:, :, 8].sum())
        takeaways.append(f"<b>Risk Range:</b> Conservative low (P10) is <b>{p10_tot:,.0f} {unit}</b>; Optimistic high (P90) is <b>{p90_tot:,.0f} {unit}</b>.")

        return {
            "title": f"Executive Demand Brief — {cfg.name} ({cfg.company})",
            "takeaways": takeaways,
            "status": "SCENARIO ACTIVE" if has_what_if else "BASELINE PROLECTION"
        }


def render_html_report(title: str, insights: dict, mode: str, horizon: int) -> str:
    """Renders a print-ready HTML executive report."""
    bullet_items = "".join([f"<li style='margin-bottom:8px;'>{t}</li>" for t in insights['takeaways']])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ScoopCast Executive Summary</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 30px; color: #0f172a; max-width: 800px; margin: 0 auto; }}
            .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; margin-bottom: 20px; }}
            .brand {{ font-size: 12px; font-weight: 700; color: #d93662; letter-spacing: 0.12em; text-transform: uppercase; }}
            .title {{ font-size: 24px; font-weight: 800; margin-top: 5px; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; background: #f1f5f9; font-size: 12px; font-weight: 600; margin-top: 10px; }}
            .section {{ margin-top: 25px; }}
            .section-title {{ font-size: 14px; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 10px; }}
            ul {{ padding-left: 20px; line-height: 1.6; font-size: 15px; }}
            .footer {{ margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 15px; font-size: 12px; color: #94a3b8; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="brand">ScoopCast Executive Intelligence</div>
            <div class="title">{title}</div>
            <div class="badge">Horizon: {horizon} Days | Status: {insights['status']}</div>
        </div>
        <div class="section">
            <div class="section-title">Automated Executive Takeaways</div>
            <ul>
                {bullet_items}
            </ul>
        </div>
        <div class="footer">
            Generated powered by Google TimesFM 3.0 Engine · ScoopCast Platform
        </div>
    </body>
    </html>
    """
    return html