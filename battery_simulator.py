import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Callable, TypedDict
import altair as alt


class BatteryConfig(TypedDict):
    capacity_wh: float
    charge_rate_w: float
    discharge_rate_w: float
    charge_efficiency: float
    discharge_efficiency: float
    min_soc: float
    max_soc: float
    initial_soc: float
    cost_per_cycle: float  # EUR per full cycle (for wear optimization)


def plot_profit_by_period(
    results: pd.DataFrame,
    period: str = "day",
    start_date: str = None,
    end_date: str = None,
) -> alt.Chart:
    """
    Create a bar chart showing profit aggregated by period.

    Args:
        results: DataFrame from BatterySimulator.run()
        period: Aggregation period - "day", "week", "month", or "year" (default: "day")
        start_date: Optional start date string 'YYYY-MM-DD' to filter data
        end_date: Optional end date string 'YYYY-MM-DD' to filter data
        title: Optional chart title (auto-generated if not provided)

    Returns:
        Altair Chart object
    """
    # Copy and prepare data
    data = results.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])

    # Apply date filtering if provided
    if start_date:
        start_dt = pd.to_datetime(start_date)
        data = data[data["datetime"] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date)
        data = data[data["datetime"] <= end_dt]

    if len(data) == 0:
        raise ValueError("No data found in the specified date range")

    # Aggregate by period
    if period == "day":
        data["period"] = data["datetime"].dt.date
        data["period"] = pd.to_datetime(data["period"])
        default_title = "Daily Profit"
        x_title = "Date"
        x_encoding = "yearmonthdate(period):T"
        tooltip_format = "%Y-%m-%d"
    elif period == "week":
        # Use Monday as the start of the week
        data["period"] = data["datetime"].dt.to_period("W-MON").dt.start_time
        default_title = "Weekly Profit"
        x_title = "Week Starting"
        x_encoding = "yearweek(period):T"
        tooltip_format = "%Y-W%W"
    elif period == "month":
        data["period"] = data["datetime"].dt.to_period("M").dt.start_time
        default_title = "Monthly Profit"
        x_title = "Month"
        x_encoding = "yearmonth(period):T"
        tooltip_format = "%Y-%m"
    elif period == "year":
        data["period"] = data["datetime"].dt.to_period("Y").dt.start_time
        default_title = "Yearly Profit"
        x_title = "Year"
        x_encoding = "year(period):T"
        tooltip_format = "%Y"
    else:
        raise ValueError(
            f"Invalid period '{period}'. Must be 'day', 'week', 'month', or 'year'"
        )

    # Calculate profit per period
    period_profit = data.groupby("period").agg({"grid_cost_eur": "sum"}).reset_index()

    period_profit["profit_eur"] = -period_profit["grid_cost_eur"]

    # Create the chart
    chart = (
        alt.Chart(period_profit)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(x_encoding, title=x_title),
            y=alt.Y("profit_eur:Q", title="Profit (€)"),
            color=alt.condition(
                alt.datum.profit_eur > 0,
                alt.value("#2ecc71"),  # Green for profit
                alt.value("#e74c3c"),  # Red for loss
            ),
            tooltip=[
                alt.Tooltip("period:T", title=x_title, format=tooltip_format),
                alt.Tooltip("profit_eur:Q", title="Profit (€)", format=".2f"),
            ],
        )
        .properties(
            title=default_title,
            width="container",
        )
    )

    return chart


def plot_hourly_battery_operation(
    results: pd.DataFrame, start_date: str, days: int = 1, title: str = None
):
    """
    Create a detailed visualization of battery operation for a date range.

    Shows:
    - Energy tariff bars (yellow=positive, green=negative)
    - Battery SOC line
    - Colored rectangles indicating charging (green) and discharging (red) periods

    Args:
        results: DataFrame from BatterySimulator.run()
        start_date: Start date string in format 'YYYY-MM-DD'
        days: Number of days to visualize (default: 1)
        title: Optional chart title (defaults to "Battery Operation - {date_range}")

    Returns:
        Altair layered Chart object
    """

    date_data = results.copy()
    date_data["datetime"] = pd.to_datetime(date_data["datetime"])

    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + timedelta(days=days)

    period_data = date_data[
        (date_data["datetime"] >= start_dt) & (date_data["datetime"] < end_dt)
    ].copy()

    if len(period_data) == 0:
        raise ValueError(
            f"No data found for date range {start_date} to {end_dt.date()}"
        )

    # Calculate dynamic width based on number of hours
    num_hours = len(period_data)
    chart_width = num_hours * 20
    chart_height = 350

    # Determine battery state (charging, discharging, idle)
    period_data["battery_state"] = "idle"
    period_data.loc[period_data["power_w"] > 0, "battery_state"] = "charging"
    period_data.loc[period_data["power_w"] < 0, "battery_state"] = "discharging"

    # Create rectangles for charging/discharging periods
    rect_data = []
    for _, row in period_data.iterrows():
        if row["battery_state"] != "idle":
            rect_data.append(
                {
                    "datetime_start": row["datetime"],
                    "datetime_end": row["datetime"] + timedelta(hours=1),
                    "state": row["battery_state"],
                }
            )

    rect_df = pd.DataFrame(rect_data)

    # Layer 1: Background rectangles for charging/discharging
    if len(rect_df) > 0:
        charging_rects = (
            alt.Chart(rect_df[rect_df["state"] == "charging"])
            .mark_rect(opacity=0.2, color="green")
            .encode(x=alt.X("datetime_start:T"), x2="datetime_end:T")
        )

        discharging_rects = (
            alt.Chart(rect_df[rect_df["state"] == "discharging"])
            .mark_rect(opacity=0.2, color="red")
            .encode(x=alt.X("datetime_start:T"), x2="datetime_end:T")
        )
    else:
        charging_rects = alt.Chart(pd.DataFrame()).mark_rect()
        discharging_rects = alt.Chart(pd.DataFrame()).mark_rect()

    # Layer 2: Price bars
    price_bars = (
        alt.Chart(period_data)
        .mark_bar(size=14, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, align="left")
        .encode(
            x=alt.X(
                "datetime:T",
                title="Date/Time",
                scale=alt.Scale(padding=25),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y("price_eur_kwh:Q", title="Price (€/kWh)", scale=alt.Scale()),
            color=alt.condition(
                alt.datum.price_eur_kwh >= 0,
                alt.value("#f1c40f"),  # Yellow for positive prices
                alt.value("#2ecc71"),  # Green for negative prices
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Date/Time"),
                alt.Tooltip("price_eur_kwh:Q", title="Price (€/kWh)", format=".4f"),
                alt.Tooltip("soc_end:Q", title="SOC", format=".0%"),
                alt.Tooltip("power_w:Q", title="Power (W)", format=".0f"),
                alt.Tooltip("battery_state:N", title="Battery State"),
            ],
        )
        .properties(width=chart_width, height=chart_height)
    )

    # Layer 3: SOC line
    soc_line = (
        alt.Chart(period_data)
        .mark_line(
            interpolate="basis",
            color="#4b8ac4",
            strokeWidth=3,
        )
        .encode(
            x=alt.X("datetime:T"),
            y=alt.Y(
                "soc_end:Q",
                title="Battery SOC (%)",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Date/Time"),
                alt.Tooltip("soc_end:Q", title="SOC", format=".1%"),
                alt.Tooltip("price_eur_kwh:Q", title="Price (€/kWh)", format=".2f"),
            ],
        )
    )

    # Layer 4: Vertical lines at midnight (excluding first and last)
    # Generate midnight timestamps for each day (skip the very first day)
    midnight_lines = []
    for day in range(1, days):
        midnight = start_dt + timedelta(days=day)
        midnight_lines.append({"midnight": midnight})

    if len(midnight_lines) > 0:
        midnight_df = pd.DataFrame(midnight_lines)
        day_separators = (
            alt.Chart(midnight_df)
            .mark_rule(
                strokeDash=[3, 3],  # Dotted line pattern
                color="lightgray",
                strokeWidth=2,
                opacity=0.3,
            )
            .encode(x="midnight:T")
        )
    else:
        day_separators = alt.Chart(pd.DataFrame()).mark_rule()

    # Combine layers with dual y-axis
    if days == 1:
        default_title = f"Battery Operation - {start_date}"
    else:
        default_title = f"Battery Operation - {start_date} to {(start_dt + timedelta(days=days-1)).date()}"

    combined = (
        alt.layer(
            charging_rects, discharging_rects, day_separators, price_bars, soc_line
        )
        .resolve_scale(y="independent")
        .properties(title=title or default_title)
    )

    return combined


def load_and_normalize_price_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load price data from JSON files and normalize to hourly intervals.

    All data is resampled to hourly intervals by averaging, regardless of
    the source interval (15-min, 60-min, or mixed).

    Args:
        data_dir: Directory containing JSON price data files

    Returns:
        DataFrame with 'datetime' and 'price' columns (hourly data)
    """
    data_path = Path(data_dir)
    all_data = []

    # Load all JSON files
    json_files = sorted(data_path.glob("*.json"))
    for json_file in json_files:
        # Load JSON
        with open(json_file, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Parse datetime (use UTC for consistency)
        df["datetime"] = pd.to_datetime(df["datum_utc"])

        # Parse price (convert comma to decimal point)
        df["price"] = df["prijs_excl_belastingen"].str.replace(",", ".").astype(float)

        # Set datetime as index and sort
        df = df[["datetime", "price"]].set_index("datetime").sort_index()

        # Resample to hourly intervals (average of all values in each hour)
        df_hourly = df.resample("1h").mean().dropna().reset_index()

        all_data.append(df_hourly)

    # Combine all years
    combined_df = (
        pd.concat(all_data, ignore_index=True)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return combined_df


class BatterySimulator:
    """
    Simulates battery operation over time with different control algorithms.

    Maintains battery state of charge, enforces physical constraints,
    and records all operations for analysis.
    """

    def __init__(self, battery_config: BatteryConfig, data_dir: str = "data") -> None:
        """
        Initialize simulator with battery specifications and load price data.

        Args:
            battery_config: dict with battery specifications:
                - capacity_wh: total capacity in Wh
                - charge_rate_w: maximum charge power in W
                - discharge_rate_w: maximum discharge power in W
                - charge_efficiency: charging efficiency (0-1)
                - discharge_efficiency: discharging efficiency (0-1)
                - min_soc: minimum state of charge (0-1)
                - max_soc: maximum state of charge (0-1)
                - initial_soc: starting state of charge (0-1)
            data_dir: Directory containing price data JSON files
        """
        self.config: BatteryConfig = battery_config

        # Load price data
        self.price_df: pd.DataFrame = load_and_normalize_price_data(data_dir)

        # Ensure price data is sorted by datetime
        self.price_df = self.price_df.sort_values("datetime").reset_index(drop=True)

    @staticmethod
    def _get_known_prices_at(
        current_time: datetime, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get prices known at current_time based on day-ahead pricing rules.

        Rules:
        - Before 13:00: Only today's prices are known (00:00 to 23:00 of current day)
        - At/after 13:00: Today's + tomorrow's prices are known (until 23:00 tomorrow)

        Args:
            current_time: datetime of current hour
            price_df: DataFrame with 'datetime' and 'price' columns

        Returns:
            DataFrame with known prices
        """
        current_date = current_time.date()
        current_hour = current_time.hour

        # Set datetime as index for easier filtering
        df = price_df.set_index("datetime")

        if current_hour < 13:
            # Before 13:00: only today's prices known
            start = datetime.combine(current_date, datetime.min.time())
            end = start + timedelta(days=1)
        else:
            # At/after 13:00: today's and tomorrow's prices known
            start = datetime.combine(current_date, datetime.min.time())
            end = start + timedelta(days=2)

        # Filter prices in the known window
        mask = (df.index >= start) & (df.index < end)
        known = df[mask].copy()

        return known.reset_index()

    def run(
        self, algorithm: Callable, verbose: bool = True, initial_state: Dict = None
    ) -> pd.DataFrame:
        """
        Run simulation with given algorithm.

        Args:
            algorithm: callable with signature:
                algorithm(current_time, battery_soc, known_prices, battery_specs, state=None) -> power_w or (power_w, new_state)
            verbose: print progress
            initial_state: optional initial state dict for stateful algorithms

        Returns:
            DataFrame with simulation results for each hour
        """
        # Initialize battery state
        soc = self.config["initial_soc"]

        # Initialize algorithm state
        algo_state = initial_state if initial_state is not None else {}

        # Results storage
        results = []

        # Loop through each hour
        for idx, row in self.price_df.iterrows():
            current_time = row["datetime"]
            current_price = row["price"]

            # Get prices known at this time
            known_prices = self._get_known_prices_at(current_time, self.price_df)

            # Call algorithm to get desired action (power in W)
            # Call algorithm - can return power_w or (power_w, new_state)
            result = algorithm(
                current_time, soc, known_prices, self.config, state=algo_state
            )

            if isinstance(result, tuple):
                desired_power_w, algo_state = result
            else:
                desired_power_w = result

            # Apply physical constraints
            actual_power_w, energy_kwh, soc_new, cost_eur = self._execute_action(
                soc, desired_power_w, current_price
            )

            # Calculate cycle fraction for this timestep
            soc_delta = abs(soc_new - soc)
            usable_range = self.config["max_soc"] - self.config["min_soc"]
            cycle_fraction = soc_delta / usable_range if usable_range > 0 else 0

            # Record results
            results.append(
                {
                    "datetime": current_time,
                    "price_eur_kwh": current_price,
                    "soc_start": soc,
                    "soc_end": soc_new,
                    "power_w": actual_power_w,
                    "energy_kwh": energy_kwh,
                    "grid_cost_eur": cost_eur,
                    "cycle_fraction": cycle_fraction,
                }
            )

            # Update battery state
            soc = soc_new

            # Progress reporting
            if verbose and idx % 1000 == 0:
                print(f"Processed {idx}/{len(self.price_df)} hours...", end="\r")

        if verbose:
            print(f"Simulation complete: {len(self.price_df)} hours processed")

        return pd.DataFrame(results)

    def _execute_action(
        self, soc: float, desired_power_w: float, price_eur_kwh: float
    ) -> Tuple[float, float, float, float]:
        """
        Execute battery action with physical constraints.

        Args:
            soc: current state of charge (0-1)
            desired_power_w: desired power (positive=charge, negative=discharge)
            price_eur_kwh: current electricity price

        Returns:
            tuple: (actual_power_w, energy_kwh, new_soc, cost_eur)
        """
        capacity_wh = self.config["capacity_wh"]

        # TODO: explore simplification of this conditional logic
        if desired_power_w > 0:
            # Charging
            actual_power_w = min(
                desired_power_w,
                self.config["charge_rate_w"],
                # Don't exceed max SoC
                (self.config["max_soc"] - soc) * capacity_wh,
            )
            energy_from_grid_kwh = (actual_power_w * 1.0) / 1000  # 1 hour
            energy_to_battery_kwh = (
                energy_from_grid_kwh * self.config["charge_efficiency"]
            )

            new_soc = soc + (energy_to_battery_kwh * 1000 / capacity_wh)
            cost_eur = energy_from_grid_kwh * price_eur_kwh  # Positive = cost

        elif desired_power_w < 0:
            # Discharging
            max_power = min(
                abs(desired_power_w),
                self.config["discharge_rate_w"],
                # Don't go below min SoC
                (soc - self.config["min_soc"]) * capacity_wh,
            )
            actual_power_w = -max_power
            energy_from_battery_kwh = (max_power * 1.0) / 1000  # 1 hour
            energy_to_grid_kwh = (
                energy_from_battery_kwh * self.config["discharge_efficiency"]
            )

            new_soc = soc - (energy_from_battery_kwh * 1000 / capacity_wh)
            cost_eur = -energy_to_grid_kwh * price_eur_kwh  # Negative = revenue

        else:
            # Idle
            actual_power_w = 0
            energy_from_grid_kwh = 0
            new_soc = soc
            cost_eur = 0

        # Ensure SoC stays within bounds (floating point safety)
        new_soc = np.clip(new_soc, self.config["min_soc"], self.config["max_soc"])

        energy_kwh = (
            energy_from_grid_kwh
            if desired_power_w > 0
            else -energy_to_grid_kwh if desired_power_w < 0 else 0
        )

        return actual_power_w, energy_kwh, new_soc, cost_eur
