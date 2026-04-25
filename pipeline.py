"""
Runs 4 local VLMs on the CARLA dataset across batch sizes [1, 3, 5] and
collects every metric for the two evaluation tables:

  Deployment Efficiency
    - Response Time      total wall-clock time for the batch
    - Avg Latency (s)    response_time / num_images
    - CPU Time (s)       CPU-only processing time (process_time)
    - RAM Usage (MB)     peak RSS increase during inference
    - Success Rate       % of images that returned non-empty valid output

  Output Quality
    - Average F1 Score       token-overlap F1 vs. a reference triple set
    - Hallucination Rate     % of generated triples using unknown entities
    - Success Calls Rate     % of model.generate() calls that completed
    - Failed Calls Rate      1 - success_calls_rate
"""

import os
import re
import gc
import csv
import time
import glob
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

import torch
# pip install psutil
import psutil 

# constants
DEFAULT_DATASET = "CARLA_DATASET_MULTI_AGENTS"
BATCH_SIZES     = [1, 3, 5]
OUTPUT_TXT      = "multi_model_results.txt"
OUTPUT_CSV      = "multi_model_metrics.csv"

RDF_PROMPT = textwrap.dedent("""
{prefixes}

You are a perception sensor for an autonomous vehicle named 'VehicleA'. Your sole function is to detect corner cases and occlusions and generate low-level observational triples based on the provided images.

CRITICAL INSTRUCTIONS:
1.  USE THE PROVIDED ONTOLOGY: You have been provided with the full AV Corner Case Ontology (AVCCO) and PROV-O ontology. This is your **only allowed vocabulary**. You must strictly use only the classes, properties, and relationships defined therein.
2.  DETECTION SCOPE: 
    - Identify all possible **corner cases** (e.g., anomalies, near-collisions, occlusions, abnormal behaviors).
    - Identify all **occlusion cases** (e.g., objects or vehicles partially/fully blocked from view).
3.  PROVENANCE IS MANDATORY:
    - Every observation **must** be attributed to this vehicle, 'VehicleA', using PROV-O properties.
    - Create one `prov:Activity` (e.g., `:vehicleA_obs_activity_1`) associated with `:VehicleA` (a `prov:Agent` and `avcco:Vehicle`).
    - Every `avcco:Observation` must be `prov:wasGeneratedBy` this activity.
4.  ONLY GENERATE OBSERVATIONS:
    - Generate only instances of `avcco:Observation` and their associated properties.
    - You are **forbidden** from generating high-level fused `avcco:Situation` instances.
5.  CONFIDENCE:
    - Assign a confidence value (0.0–1.0) to each observation using `avcco:hasConfidenceScore`.
6.  DIRECTIONAL CONTEXT (NEW):
    - Infer the approximate **direction or orientation** of observed entities relative to 'VehicleA' 
      (e.g., `avcco:hasDirection "north"`, `"east"`, `"south"`, `"west"`), based on image cues such as road alignment, shadows, and map compass overlays.
7.  VEHICLE RELATIVE POSITION:
    - If possible, include relative spatial terms such as `"front"`, `"rear"`, `"left"`, `"right"` using the property `avcco:hasRelativePosition`.
8.  ONTOLOGY COMPLIANCE:
    - Use only classes and properties defined in the AVCCO and PROV-O ontologies. Do not invent new ones.
9.  OUTPUT FORMAT:
    - Return only valid RDF triples in Turtle syntax using the provided prefixes.

Ontology reference:
{ontology_prompt}
""").strip()

# Entities expected in autonomous driving scenes
KNOWN_ENTITIES = {
    "car", "vehicle", "bus", "truck", "motorcycle", "bicycle", "pedestrian",
    "person", "building", "road", "street", "sidewalk", "sign", "traffic",
    "light", "tree", "sky", "lane", "intersection", "crosswalk", "barrier",
    "fence", "pole", "window", "wheel", "door", "roof", "ground", "pavement",
}

