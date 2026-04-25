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
import time
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

prefixes = """
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

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


# required metrics for complete testing

def _ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1_048_576


def _clean_turtle(raw: str) -> str:
    if "```turtle" in raw:
        m = re.search(r"```turtle(.+?)```", raw, re.DOTALL)
        raw = m.group(1).strip() if m else raw.strip()
    elif "```" in raw:
        m = re.search(r"```(.+?)```", raw, re.DOTALL)
        raw = m.group(1).strip() if m else raw.strip()
    return raw.replace("ex/", "ex:").strip()


def _is_valid_turtle(text: str) -> bool:
    return bool(text) and ("." in text) and ("ex:" in text or "@prefix" in text)


def _token_set(text: str) -> set:
    return set(re.sub(r"[^a-z0-9]", " ", text.lower()).split())


def compute_f1(generated: str, reference: str) -> float:
    """F1 score between generated output and reference Turtle."""
    gen = _token_set(generated)
    ref = _token_set(reference)
    if not gen or not ref:
        return 0.0
    common = gen & ref
    if not common:
        return 0.0
    precision = len(common) / len(gen)
    recall    = len(common) / len(ref)
    return 2 * precision * recall / (precision + recall)


def compute_hallucination_rate(generated: str) -> float:
    """Fraction of ex: entity tokens not in KNOWN_ENTITIES."""
    entities = re.findall(r"ex:([A-Za-z]+)", generated, re.IGNORECASE)
    if not entities:
        return 0.0
    unknown = [e for e in entities if e.lower() not in KNOWN_ENTITIES]
    return len(unknown) / len(entities)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rgb_only(paths):
    return [p for p in paths if Path(p).suffix.lower() in (".png", ".jpg", ".jpeg")]


def _log(msg, fh=None):
    print(msg)
    if fh:
        fh.write(msg + "\n")
        fh.flush()
