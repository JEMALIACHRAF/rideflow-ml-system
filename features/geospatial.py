"""
Geospatial feature engineering.
H3 hexagonal indexing, zone clustering, POI proximity.
"""
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# Paris zone adjacency (arrondissements sharing a border)
ZONE_ADJACENCY = {
    "1":  ["2", "4"],
    "2":  ["1", "3", "9", "10"],
    "3":  ["2", "4", "10", "11"],
    "4":  ["1", "3", "5", "12"],
    "5":  ["4", "6", "13"],
    "6":  ["5", "7", "14", "15"],
    "7":  ["6", "8", "15"],
    "8":  ["7", "9", "17"],
    "9":  ["2", "8", "10", "17", "18"],
    "10": ["2", "3", "9", "11", "18", "19"],
    "11": ["3", "4", "10", "12", "19", "20"],
    "12": ["4", "5", "11", "13", "20"],
    "13": ["5", "12", "14"],
    "14": ["6", "13", "15"],
    "15": ["6", "7", "14", "16"],
    "16": ["7", "8", "15", "17"],
    "17": ["8", "9", "16", "18"],
    "18": ["9", "10", "17", "19"],
    "19": ["10", "11", "18", "20"],
    "20": ["11", "12", "19"],
}

# Key POIs that attract / repel rides
POI_ZONES = {
    "cdg_airport":    ("18", 40.0),
    "orly_airport":   ("13", 35.0),
    "gare_du_nord":   ("10", 25.0),
    "eiffel_tower":   ("7",  20.0),
    "louvre":         ("1",  15.0),
    "bercy_arena":    ("12", 30.0),
    "parc_princes":   ("16", 35.0),
    "la_defense":     ("17", 20.0),
}

# Zone clusters (central / mid-ring / outer)
ZONE_CLUSTER_MAP = {
    "1": "central", "2": "central", "3": "central", "4": "central",
    "5": "central", "6": "central", "7": "central", "8": "central",
    "9": "mid",  "10": "mid", "11": "mid", "12": "mid",
    "13": "mid", "14": "mid", "15": "mid", "16": "mid",
    "17": "outer", "18": "outer", "19": "outer", "20": "outer",
}


def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode zone and add cluster, adjacency demand."""
    df = df.copy()
    le = LabelEncoder()
    df["zone_encoded"] = le.fit_transform(df["zone"].astype(str))
    df["zone_cluster"] = df["zone"].map(ZONE_CLUSTER_MAP)
    cluster_dummies = pd.get_dummies(df["zone_cluster"], prefix="cluster")
    df = pd.concat([df, cluster_dummies], axis=1)
    return df


def add_adjacency_demand(df: pd.DataFrame, target: str = "demand") -> pd.DataFrame:
    """
    For each zone at each timestamp, compute the mean demand of neighbouring zones.
    Captures spatial spillover effects.
    """
    df = df.copy()
    demand_pivot = df.pivot_table(index="timestamp", columns="zone", values=target, aggfunc="mean")

    adj_demand = {}
    for zone, neighbors in ZONE_ADJACENCY.items():
        valid_neighbors = [n for n in neighbors if n in demand_pivot.columns]
        if valid_neighbors:
            adj_demand[zone] = demand_pivot[valid_neighbors].mean(axis=1)

    adj_df = pd.DataFrame(adj_demand).reset_index()
    adj_df = adj_df.melt(id_vars="timestamp", var_name="zone", value_name="adj_zone_demand")
    df = df.merge(adj_df, on=["timestamp", "zone"], how="left")
    logger.debug("Added adjacency demand features")
    return df


def add_poi_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary/proximity features for zones near major POIs.
    Each POI has a primary zone and a gravitational weight.
    """
    df = df.copy()
    for poi_name, (poi_zone, weight) in POI_ZONES.items():
        # Primary zone gets full weight, adjacent zones get partial
        adj = ZONE_ADJACENCY.get(poi_zone, [])
        df[f"poi_{poi_name}"] = df["zone"].apply(
            lambda z: weight if z == poi_zone else (weight * 0.4 if z in adj else 0.0)
        )
    logger.debug(f"Added {len(POI_ZONES)} POI proximity features")
    return df


def add_spatial_demand_ratio(df: pd.DataFrame, target: str = "demand") -> pd.DataFrame:
    """
    Zone demand as a fraction of total Paris demand at that hour.
    Captures relative hotspot intensity.
    """
    df = df.copy()
    total = df.groupby("timestamp")[target].transform("sum")
    df["demand_share"] = df[target] / (total + 1e-6)
    df["demand_rank"]  = df.groupby("timestamp")[target].rank(ascending=False, method="min")
    return df


def build_geospatial_features(df: pd.DataFrame, target: str = "demand") -> pd.DataFrame:
    """Full geospatial feature pipeline."""
    df = add_zone_features(df)
    df = add_adjacency_demand(df, target)
    df = add_poi_proximity_features(df)
    df = add_spatial_demand_ratio(df, target)
    return df
