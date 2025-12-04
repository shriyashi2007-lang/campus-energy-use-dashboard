#Name: shriyashi
#roll no.:2501420023

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt



#CONFIG
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CLEANED_DATA_PATH = OUTPUT_DIR / "cleaned_energy_data.csv"
BUILDING_SUMMARY_PATH = OUTPUT_DIR / "building_summary.csv"
SUMMARY_TXT_PATH = OUTPUT_DIR / "summary.txt"
DASHBOARD_IMG_PATH = OUTPUT_DIR / "dashboard.png"


#OOP MODELING 

@dataclass
class MeterReading:
    timestamp: pd.Timestamp
    kwh: float


class Building:
    def __init__(self, name: str):
        self.name = name
        self.meter_readings: List[MeterReading] = []

    def add_reading(self, timestamp, kwh):
        self.meter_readings.append(MeterReading(pd.to_datetime(timestamp), float(kwh)))

    def calculate_total_consumption(self) -> float:
        return sum(r.kwh for r in self.meter_readings)

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "building": [self.name] * len(self.meter_readings),
            "timestamp": [r.timestamp for r in self.meter_readings],
            "kwh": [r.kwh for r in self.meter_readings],
        }
        return pd.DataFrame(data)

    def generate_report(self) -> str:
        total = self.calculate_total_consumption()
        return f"Building: {self.name}, Total consumption: {total:.2f} kWh"


class BuildingManager:
    def __init__(self):
        self.buildings: Dict[str, Building] = {}

    def get_or_create_building(self, name: str) -> Building:
        if name not in self.buildings:
            self.buildings[name] = Building(name)
        return self.buildings[name]

    def load_from_dataframe(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            bname = row["building"]
            ts = row["timestamp"]
            kwh = row["kwh"]
            building = self.get_or_create_building(bname)
            building.add_reading(ts, kwh)

    def generate_reports(self) -> List[str]:
        return [b.generate_report() for b in self.buildings.values()]


# DATA INGESTION & VALIDATION 

def load_and_merge_data(data_dir: Path) -> pd.DataFrame:
    all_rows = []
    log_messages = []

    for csv_path in data_dir.glob("*.csv"):
        try:
            # Try to infer building name from filename if not in file
            inferred_building = csv_path.stem  # e.g., "building_a_jan"
            try:
                df = pd.read_csv(csv_path, on_bad_lines="skip")
            except TypeError:
                df = pd.read_csv(csv_path, error_bad_lines=False)

            # Ensure required columns exist or add them
            if "building" not in df.columns:
                df["building"] = inferred_building

            if "timestamp" not in df.columns or "kwh" not in df.columns:
                log_messages.append(f"Skipping {csv_path.name}: missing 'timestamp' or 'kwh' column")
                continue

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "kwh"])

            # Optional: add month metadata
            df["month"] = df["timestamp"].dt.to_period("M").astype(str)

            all_rows.append(df)

        except FileNotFoundError:
            log_messages.append(f"File not found: {csv_path}")
        except Exception as e:
            log_messages.append(f"Error reading {csv_path.name}: {e}")

    if not all_rows:
        raise ValueError("No valid CSV files were loaded from data/")

    df_combined = pd.concat(all_rows, ignore_index=True)

    # Simple log printing (could also write to a log file)
    if log_messages:
        print("Ingestion/Validation log:")
        for msg in log_messages:
            print(" -", msg)

    return df_combined


# AGGREGATION LOGIC 

def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("timestamp").sort_index()
    daily_totals = df.resample("D")["kwh"].sum().reset_index()
    daily_totals.rename(columns={"kwh": "daily_kwh"}, inplace=True)
    return daily_totals


def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("timestamp").sort_index()
    weekly_totals = df.resample("W")["kwh"].sum().reset_index()
    weekly_totals.rename(columns={"kwh": "weekly_kwh"}, inplace=True)
    return weekly_totals


def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("building")["kwh"].agg(
        total="sum",
        mean="mean",
        min="min",
        max="max"
    ).reset_index()
    return summary


def building_weekly_average(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_building = df.groupby(["building", "week"])["kwh"].sum().reset_index()
    avg_weekly = weekly_building.groupby("building")["kwh"].mean().reset_index()
    avg_weekly.rename(columns={"kwh": "avg_weekly_kwh"}, inplace=True)
    return avg_weekly


def peak_hour_consumption(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby(["building", "hour"])["kwh"].sum().reset_index()
    return hourly


# VISUAL OUTPUT WITH MATPLOTLIB 

def create_dashboard_figure(df: pd.DataFrame, daily_totals: pd.DataFrame,
                            avg_weekly: pd.DataFrame, hourly: pd.DataFrame,
                            save_path: Path):
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Trend line – daily consumption over time for all buildings
    ax1 = axes[0]
    df_daily_building = df.copy()
    df_daily_building["date"] = df_daily_building["timestamp"].dt.date
    daily_building = df_daily_building.groupby(["date", "building"])["kwh"].sum().reset_index()
    for building, sub in daily_building.groupby("building"):
        ax1.plot(sub["date"], sub["kwh"], marker="o", label=building)
    ax1.set_title("Daily Consumption by Building")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("kWh")
    ax1.legend(fontsize=8)

    # 2) Bar chart – compare average weekly usage across buildings
    ax2 = axes[1]
    ax2.bar(avg_weekly["building"], avg_weekly["avg_weekly_kwh"], color="tab:blue")
    ax2.set_title("Average Weekly Consumption")
    ax2.set_xlabel("Building")
    ax2.set_ylabel("kWh (avg weekly)")
    ax2.tick_params(axis="x", rotation=45)

    # 3) Scatter plot – peak-hour consumption vs time/building
    ax3 = axes[2]
    # take top N hours per building by consumption
    top_hours = hourly.sort_values("kwh", ascending=False).groupby("building").head(5)
    scatter = ax3.scatter(top_hours["hour"], top_hours["kwh"],
                          c=top_hours["hour"], cmap="viridis")
    for _, row in top_hours.iterrows():
        ax3.annotate(row["building"], (row["hour"], row["kwh"]), fontsize=6, alpha=0.7)
    ax3.set_title("Peak Hour Consumption (Top 5 per Building)")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Total kWh")
    fig.colorbar(scatter, ax=ax3, label="Hour")

    fig.suptitle("Campus Energy Dashboard", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


#  PERSISTENCE + EXECUTIVE SUMMARY 

def generate_summary_text(df: pd.DataFrame, building_summary: pd.DataFrame,
                          daily_totals: pd.DataFrame, hourly: pd.DataFrame) -> str:
    total_campus = df["kwh"].sum()

    # Highest  consuming building
    highest_row = building_summary.sort_values("total", ascending=False).iloc[0]
    highest_building = highest_row["building"]
    highest_building_total = highest_row["total"]

    # Peak load time (hour with highest total kWh across campus)
    hourly_total = df.copy()
    hourly_total["hour"] = hourly_total["timestamp"].dt.hour
    by_hour = hourly_total.groupby("hour")["kwh"].sum().reset_index()
    peak_hour_row = by_hour.sort_values("kwh", ascending=False).iloc[0]
    peak_hour = int(peak_hour_row["hour"])
    peak_hour_kwh = peak_hour_row["kwh"]

    #simple daily/weekly trend descriptions
    first_day = daily_totals["timestamp"].min().date()
    last_day = daily_totals["timestamp"].max().date()
    max_daily = daily_totals.sort_values("daily_kwh", ascending=False).iloc[0]

    summary_lines = [
        "Campus Energy Consumption Summary",
        "--------------------------------",
        f"Total campus consumption: {total_campus:.2f} kWh",
        f"Highest consuming building: {highest_building} "
        f"({highest_building_total:.2f} kWh)",
        f"Peak load hour (campus-wide): {peak_hour}:00 "
        f"with {peak_hour_kwh:.2f} kWh",
        f"Data range: {first_day} to {last_day}",
        (f"Maximum daily campus consumption of "
         f"{max_daily['daily_kwh']:.2f} kWh on {max_daily['timestamp'].date()}")
    ]
    return "\n".join(summary_lines)


def main():
    #1) Ingest and validate
    df = load_and_merge_data(DATA_DIR)

    #Ensure correct dtypes
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["kwh"])
    df = df.sort_values("timestamp")

    # Savecleaned combined data
    df.to_csv(CLEANED_DATA_PATH, index=False)

    #2) Aggregations
    daily_totals = calculate_daily_totals(df)
    weekly_totals = calculate_weekly_aggregates(df)
    building_summary = building_wise_summary(df)
    avg_weekly = building_weekly_average(df)
    hourly = peak_hour_consumption(df)

    #Save building summary
    building_summary.to_csv(BUILDING_SUMMARY_PATH, index=False)

    #3) OOP: Load into BuildingManager and show reports
    manager = BuildingManager()
    manager.load_from_dataframe(df)
    reports = manager.generate_reports()
    print("Per-building reports:")
    for rep in reports:
        print(rep)

    # 4)Dashboard figure
    create_dashboard_figure(df, daily_totals, avg_weekly, hourly, DASHBOARD_IMG_PATH)
    print(f"Dashboard saved to: {DASHBOARD_IMG_PATH}")

    #5) Executive summary
    summary_text = generate_summary_text(df, building_summary, daily_totals, hourly)
    SUMMARY_TXT_PATH.write_text(summary_text, encoding="utf-8")
    print("\nExecutive Summary:")
    print(summary_text)
    print(f"\nSummary written to: {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()