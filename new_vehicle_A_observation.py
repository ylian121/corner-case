# This file contains the logic for the process
# Note: We need to implement a mechanism for continuous arrival of new images.
# Path of the images folder
#   images
#       - scenarios
#            - RGB
#            - LIDAR

from PIL import Image
import os
import re
import glob
import gc
import torch
from rdflib import Graph, RDF, RDFS, OWL
import weather_classifier_inference as uciclassifier

# import 4 models
# import moondream
# import glm
# import smolvlm
import qwen

import time
import psutil

import json
from datetime import datetime, timezone

METRICS_FILE = "output/qwen_metrics.jsonl"

def log_metrics(scenario, weather, loop, img_path, prompt_text, raw_output, latency, status, error=None):
    os.makedirs("output", exist_ok=True)
    record = {
        "scenario": scenario,
        "weather": weather,
        "loop": loop,
        "batch_size": 1,
        "model": "Qwen2-VL-2B",
        "vehicle": "A",
        "image": os.path.basename(img_path),
        "latency_s": round(latency, 4),
        "input_bytes": len(prompt_text.encode("utf-8")),
        "output_bytes": len(raw_output.encode("utf-8")) if raw_output else 0,
        "prompt_tokens": len(prompt_text) // 4,       # rough estimate
        "completion_tokens": len(raw_output) // 4 if raw_output else 0,
        "success": status == "success",
        "error": error,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))

class BenchmarkTracker:
    def __init__(self, ontology_path):
        self.ontology = Graph().parse(ontology_path, format="turtle")
        self.ontology_predicates = set(self.ontology.predicates())
        
    def calculate_quality(self, generated_graph, ground_truth_graph=None):
        """Calculates Ontology Compliance and F1 """
        gen_triples = set(generated_graph)
        
        # 1. Ontology Compliance (How many predicates are actually in your TTL file?)
        if len(gen_triples) == 0:
            compliance = 0.0
        else:
            valid_triples = sum(1 for _, p, _ in gen_triples if p in self.ontology_predicates)
            compliance = (valid_triples / len(gen_triples)) * 100

        # 2. F1 Score
        f1_score = 0.0
        if ground_truth_graph:
            gt_triples = set(ground_truth_graph)
            tp = len(gen_triples.intersection(gt_triples))
            fp = len(gen_triples - gt_triples)
            fn = len(gt_triples - gen_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 3. Hallucination Rate
        hallucination_rate = 100.0 - compliance
        
        return round(f1_score, 3), round(hallucination_rate, 2), round(compliance, 2)

    def get_system_usage(self):
        """Capture RAM and VRAM usage."""
        ram_usage = psutil.virtual_memory().percent
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / 1024**2 # Convert to MB
        return ram_usage, round(vram_usage, 2)

    def summarize_scenario(self, model_name, batch_size, loop_data):
        """Formats the data into your requested table structure."""
        total_loops = len(loop_data)
        successes = sum(1 for d in loop_data if d['status'] == 'success')
        
        summary = {
            "Model": model_name,
            "Batch Size": batch_size,
            "Avg Latency (s)": round(sum(d['latency'] for d in loop_data) / total_loops, 2),
            "CPU Time (s)": round(sum(d['cpu_time'] for d in loop_data), 2),
            "RAM Usage (%)": f"{sum(d['ram'] for d in loop_data) / total_loops}%",
            "Success Rate": f"{(successes/total_loops)*100}%",
            "Avg F1 Score": round(sum(d['f1'] for d in loop_data) / total_loops, 3),
            "Hallucination Rate": f"{round(sum(d['halluc'] for d in loop_data) / total_loops, 2)}%",
            "Ontology Compliance": f"{round(sum(d['comp'] for d in loop_data) / total_loops, 2)}%",
            "Success Calls": successes,
            "Failed Calls": total_loops - successes
        }
        return summary


MODELS = {
    "Qwen": qwen,
    # "GLM-OCR": glm,
    # "Moondream": moondream,
    # "SmolVLM": smolvlm,
}
'''
    "GLM-OCR": glm,
    "SmolVLM": smolvlm,
    "Qwen": qwen,
'''

BATCH_SIZES = [1]



# === Function to extract ontology summary as prompt ===
def extract_ontology_prompt(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")

    class_lines = ["Ontology Classes (and Hierarchy):"]
    for s in g.subjects(RDF.type, OWL.Class):
        label = g.value(s, RDFS.label)
        comment = g.value(s, RDFS.comment)
        subclass_of = g.value(s, RDFS.subClassOf)
        class_name = s.split("#")[-1] if "#" in s else s
        superclass = (
            subclass_of.split("#")[-1]
            if subclass_of and "#" in subclass_of
            else subclass_of
        )
        line = f"- {class_name}"
        if superclass:
            line += f" (subclass of {superclass})"
        if label:
            line += f": {label}"
        if comment:
            line += f"\n  {comment}"
        class_lines.append(line)

    property_lines = ["\nOntology Properties:"]
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_name = s.split("#")[-1] if "#" in s else s
        property_lines.append(f"- {prop_name}")

    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        prop_name = s.split("#")[-1] if "#" in s else s
        property_lines.append(f"- {prop_name}")

    return "\n".join(class_lines + property_lines)


# === Function to extract confidence scores ===
def compute_avg_confidence_score(triples):
    confidence_scores = []
    current_subject = None
    for line in triples.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("@prefix") or line.startswith("#"):
            continue

        if "hasConfidenceScore" not in line:
            continue

        # Detect subject line
        # avcco:hasConfidenceScore "0.95"^^xsd:float ;
        # avcco:hasConfidenceScore "0.95"^^xsd:float .
        if line.endswith(";") or line.endswith("."):
            parts = line.split(" ", 2)
            if len(parts) == 3:
                s, p, o = parts
                s = s.strip().split(":")[-1]
                p = p.strip().split("^^")[0].strip('"')

                # Check for confidence score
                if s == "hasConfidenceScore":
                    try:
                        confidence_scores.append(float(p))
                    except ValueError:
                        pass
            current_subject = parts[0].strip().split(":")[-1]
        else:
            # Handle multiline continuation for the same subject
            parts = line.split(" ", 1)
            if len(parts) == 2 and current_subject:
                p, o = parts
                p = p.strip().split(":")[-1]
                o = o.strip("<>").strip('"')

                # Check for confidence score
                if p == "hasConfidenceScore":
                    try:
                        confidence_scores.append(float(o))
                    except ValueError:
                        pass

    # Calculate average LVLM confidence score
    return sum(confidence_scores) / len(confidence_scores) if confidence_scores else None


# === Function to average weighted score by using UCI classifier ===
# NOTE: Use the list of images we used to get the triples from the LLM
def compute_avg_classifier_score(image_paths):
    classifier = uciclassifier.WeatherClassifier()
    class_weights = {
        "Day": 1.00,
        "Night": 1.25,
        "Fog": 1.30
    }
    # The maximum weight should be less than 1.17

    total_weighted_score = 0.0
    valid_image_count = 0

    for image_path in image_paths:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                label = classifier.predict_image(image_path=image_path).strip()
                if label not in class_weights:
                    continue
                total_weighted_score += class_weights[label]
                valid_image_count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    return total_weighted_score / valid_image_count if valid_image_count > 0 else 0.0

import re

def _filter_turtle_lines(text: str) -> str:
    text = re.sub(r"b'(.*?)'", r"\1", text)
    text = re.sub(r'b"(.*?)"', r"\1", text)
    valid_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            valid_lines.append("")
            continue
        if (
            stripped.startswith("@")
            or stripped.startswith("PREFIX")
            or stripped.startswith("<")
            or stripped.startswith("#")
            or re.search(r'\w:\w', stripped)
            or re.fullmatch(r'[.,;{}]+', stripped)
        ):
            valid_lines.append(line)
    return "\n".join(valid_lines).strip()


def get_triples_from_local_model(model_mod, image_paths, prompt_text):
    clean_output = ""
    try:
        rgb_image_paths = [
            p for p in image_paths
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not rgb_image_paths:
            return ""

        # Single call — image + prompt together
        raw_output = model_mod.run_inference(rgb_image_paths, prompt_text)
        print(f"  [RAW]: {repr(raw_output[:300])}")

        if not raw_output:
            return ""

        # Extract and clean
        if "```turtle" in raw_output:
            match = re.search(r'```turtle(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        elif "```" in raw_output:
            match = re.search(r'```(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        else:
            clean_output = raw_output.strip()

        clean_output = _filter_turtle_lines(clean_output)
        return clean_output.replace("ex/", "ex:")

    except Exception as e:
        print(f"  [ERROR] inference failed: {e}")
        return ""

'''
def get_triples_from_local_model(model_mod, image_paths, prompt_text):
    clean_output = ""
    try:
        # 1. Filter valid images, slice to 5 for now
        rgb_image_paths = [
            p for p in image_paths
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ][:5]
        if not rgb_image_paths:
            return ""

        # 2. Qwen2-VL handles the full list in one call
        raw_output = model_mod.run_inference(rgb_image_paths, prompt_text)
        print(f"  [RAW]: {repr(raw_output[:300])}")

        if not raw_output:
            return ""

        # 3. Extract Turtle block
        if "```turtle" in raw_output:
            match = re.search(r'```turtle(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        elif "```" in raw_output:
            match = re.search(r'```(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        else:
            clean_output = raw_output.strip()

        clean_output = _filter_turtle_lines(clean_output)
        return clean_output.replace("ex/", "ex:")

    except Exception as e:
        print(f"  [ERROR] Qwen2-VL inference failed: {e}")
        return ""

'''

'''
from PIL import Image
import re
import os

# Shrink images before sending to the model
MAX_IMAGE_DIM = 512  # Cap longest side at 512px for speed; raise to 768/1024 later

def _resize_image(path: str, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def get_triples_from_local_model(model_mod, image_paths, prompt_text):
    clean_output = ""
    try:
        # 1. Filter valid images, take first 5 for smoke test
        rgb_image_paths = [
            p for p in image_paths
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ][:5]
        if not rgb_image_paths:
            return ""

        # 2. run_inference handles _load() internally — don't call model_mod.processor directly
        raw_output = model_mod.run_inference(rgb_image_paths, prompt_text)
        print(f"  [RAW]: {repr(raw_output[:300])}")

        if not raw_output:
            return ""

        # 3. Extract Turtle block
        if "```turtle" in raw_output:
            match = re.search(r'```turtle(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        elif "```" in raw_output:
            match = re.search(r'```(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        else:
            clean_output = raw_output.strip()

        clean_output = _filter_turtle_lines(clean_output)
        return clean_output.replace("ex/", "ex:")

    except Exception as e:
        print(f"  [ERROR] GLM-OCR inference failed: {e}")
        return ""

import re

def _filter_turtle_lines(text: str) -> str:
    """
    Keep only lines that are valid Turtle RDF syntax.
    Drops plain-English prose the model sometimes injects.
    """
    # Strip Python byte-string artifacts: b'...' or b"..."
    text = re.sub(r"b'(.*?)'", r"\1", text)
    text = re.sub(r'b"(.*?)"', r"\1", text)

    valid_lines = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            valid_lines.append("")
            continue

        if (
            stripped.startswith("@")              # @prefix, @base
            or stripped.startswith("PREFIX")      # SPARQL-style prefix
            or stripped.startswith("<")           # URI subject: <http://...>
            or stripped.startswith("#")           # comment
            or re.search(r'\w:\w', stripped)      # prefixed name: ex:Foo, rdf:type
                                                  # "Entity: Foo" has space → no match ✓
            or re.fullmatch(r'[.,;{}]+', stripped) # punctuation-only lines
        ):
            valid_lines.append(line)
        # else: plain English prose → silently dropped

    return "\n".join(valid_lines).strip()


def get_triples_from_local_model(model_mod, image_paths, prompt_text):
    """
    Standardized bridge for SmolVLM.
    SmolVLM handles the list of paths internally, so we just pass them through.
    """
    clean_output = ""

    try:
        # 1. Filter for valid images
        rgb_image_paths = [
            p for p in image_paths
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not rgb_image_paths:
            return ""

        # 2. Call SmolVLM's run_inference
        raw_output = model_mod.run_inference(rgb_image_paths, prompt_text)

        if not raw_output:
            return ""

        # 3. Extract triples from markdown code blocks
        if "```turtle" in raw_output:
            match = re.search(r'```turtle(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        elif "```" in raw_output:
            match = re.search(r'```(.+?)```', raw_output, re.DOTALL)
            clean_output = match.group(1).strip() if match else raw_output
        else:
            clean_output = raw_output.strip()

        # 4. Drop any prose lines the model injected into the Turtle block
        clean_output = _filter_turtle_lines(clean_output)

        # 5. Ontology cleanup
        return clean_output.replace("ex/", "ex:")

    except Exception as e:
        print(f"Error calling SmolVLM: {e}")
        return ""


def get_triples_from_local_model(model_mod, image_paths, prompt_text):


    # Standardized bridge to call local models and clean the Turtle output.

    # triples = open("raw_output.ttl", "w")
    # triples_file.write(raw_output.strip())
    # triples_file.close()

    # Extract triples from markdown code block if present
    # TODO: Handle other issues:
    # 1. ex/ to ex:
    # 2. Remove any text before or after the triples
    # 3. Handle both ```turtle and ```
    # 4. Handle spaces before comments.
    try:
        # only RGB images 
        rgb_image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Call the local model's standardized inference function
        raw_output = model_mod.run_inference(rgb_image_paths, prompt_text)

        if "moondream" in model_mod.__name__.lower():
            from PIL import Image
            img = Image.open(rgb_image_paths[0])
            raw_output = model_mod.run_inference(img, prompt_text)
        else:
            # Qwen and others handle the list of paths
            raw_output = model_mod.run_inference(rgb_image_paths, prompt_text)

        if not raw_output:
            return ""

        if "```turtle" in raw_output:
            triples = re.search('```turtle(.+?)```', raw_output, re.DOTALL)
            clean_output = triples.group(1).strip() if triples else raw_output
        elif "```" in raw_output:
            triples = re.search('```(.+?)```', raw_output, re.DOTALL)
            clean_output = triples.group(1).strip() if triples else raw_output
        else:
            clean_output = raw_output.strip()
            
        return clean_output.replace("ex/", "ex:") if clean_output else ""

    except Exception as e:
        print(f"Error calling {model_mod.__name__}: {e}")
        return ""
'''



# === Paths ===
scenarios_folder = r"CARLA_DATASET_MULTI_AGENTS"
ttl_path = r"avcc_with_reasoning_no_shacl.ttl"

main_graph = Graph()

# === Ontology & prompt setup ===
ontology_prompt = extract_ontology_prompt(ttl_path)


prompt = """You are an autonomous vehicle perception system analyzing a dashcam image.

Detect and output ALL of the following as Turtle RDF triples:

1. All visible vehicles (cars, buses, trucks) - type, color, position
2. Weather conditions - fog, rain, clear
3. Occlusions - any vehicle partially or fully blocked from view
4. Corner cases - unusual situations, near collisions, abnormal behavior
5. Road conditions and visibility

Use these prefixes:
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex: <http://cornercase.org/instances#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

Requirements:
- Every entity gets a unique ID (ex:Car1, ex:Bus1, ex:Obs1 etc.)
- Every observation needs avcco:hasConfidenceScore (0.0-1.0)
- Every observation needs prov:wasGeneratedBy ex:vehicleA_activity_1
- Include avcco:hasRelativePosition ("front"/"rear"/"left"/"right")
- If a vehicle is occluded, link it: ex:Obs1 avcco:isOccludedBy ex:Bus1

Start with ```turtle and end with ```. No prose. No explanation.

Example:
```turtle
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex: <http://cornercase.org/instances#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:vehicleA_activity_1 a prov:Activity ;
    prov:wasAssociatedWith ex:VehicleA .

ex:VehicleA a prov:Agent .

ex:Obs1 a avcco:Observation ;
    prov:wasGeneratedBy ex:vehicleA_activity_1 ;
    avcco:hasConfidenceScore 0.85 ;
    avcco:hasRelativePosition "front" ;
    avcco:refersTo ex:Car1 .

ex:Car1 a avcco:Car ;
    avcco:hasColor "white" ;
    avcco:isOccludedBy ex:Bus1 .

ex:Bus1 a avcco:Bus ;
    avcco:hasRelativePosition "front" .

ex:WeatherObs1 a avcco:Observation ;
    prov:wasGeneratedBy ex:vehicleA_activity_1 ;
    avcco:hasConfidenceScore 0.9 ;
    avcco:hasWeatherCondition avcco:Fog .
```
"""

'''
print(f"Ontology prompt length: {len(ontology_prompt)}")
print(ontology_prompt[:300])

prefixes = """
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

perception_prompt = """Look at this driving scene image. List every object you can see.
For each object write one line: OBJECT | COLOR | POSITION (front/rear/left/right) | VISIBILITY (clear/partial/blocked)
Example:
Car | white | front | partial
Bus | yellow | left | clear
Road | gray | front | clear

Only output the list. No sentences."""


def make_turtle_prompt(perception_output):
    return f"""Convert this object list to Turtle RDF triples.

Object list:
{perception_output}

Use exactly these prefixes:
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex: <http://cornercase.org/instances#> .
@prefix prov: <http://www.w3.org/ns/prov#> .

Rules:
- Every object becomes an avcco:Observation
- Use avcco:hasRelativePosition for position
- Use avcco:hasConfidenceScore (0.0-1.0) based on visibility
- Every observation must have prov:wasGeneratedBy ex:vehicleA_activity_1

Start with ```turtle and end with ```. Nothing else.

Example output:
```turtle
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex: <http://cornercase.org/instances#> .
@prefix prov: <http://www.w3.org/ns/prov#> .

ex:vehicleA_activity_1 a prov:Activity ;
    prov:wasAssociatedWith ex:VehicleA .

ex:Obs1 a avcco:Observation ;
    prov:wasGeneratedBy ex:vehicleA_activity_1 ;
    avcco:hasRelativePosition "front" ;
    avcco:hasConfidenceScore 0.7 .
```
"""
'''
'''
prompt = f"""
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
"""
'''


tracker = BenchmarkTracker(ttl_path)

# For each scenario in the root
for model_name, model_mod in MODELS.items():

    model_stats = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nRunning model: {model_name}")
    print("\n")

    model_limit = getattr(model_mod, "MAX_BATCH_SIZE", 1)

    for scenario in os.listdir(scenarios_folder):
        scenario_folder = os.path.join(scenarios_folder, scenario)
        if not os.path.isdir(scenario_folder):
            continue

        # For each weather in the scenario
        for weather in os.listdir(scenario_folder):
            main_graph = Graph()
            loop = 0  # first loop images start from 0 and 0 lidar images
            adjusted_score = 0.0

            weather_folder = os.path.join(scenario_folder, weather)
            if not os.path.isdir(weather_folder):
                continue

            vehicle_folder = os.path.join(weather_folder, "A")
            if not os.path.isdir(vehicle_folder):
                continue

            print(f"Processing scenario: {scenario}, weather: {weather}, vehicle: A")

            rgbs_folder = os.path.join(vehicle_folder, "rgb")
            if not os.path.isdir(rgbs_folder):  # Also check the folder i not empty
                continue


            # Use 5 RGB from loop to get the confidence score
            rgb_images = sorted(
                glob.glob(os.path.join(rgbs_folder, "*.png")) +
                glob.glob(os.path.join(rgbs_folder, "*.jpg"))
            )

            while (loop * model_limit) < len(rgb_images):
                print(f"Loop: {loop}, RGB images: {len(rgb_images)}")

                start_idx = loop * model_limit
                selected_images = rgb_images[start_idx : start_idx + model_limit]

                t0 = time.time()
                cpu0 = time.process_time()
                ram, vram = tracker.get_system_usage()

                # --- FIX START ---
                status = "failed" # Set a default status so the code doesn't crash later
                temp = Graph()
                loop_raw_triples = ""

                try:
                    for img_path in selected_images:
                        print(f"  -> Processing: {os.path.basename(img_path)}")
                        
                        t_img = time.time()  # ADD: per-image timer
                        img_error = None
                        img_status = "failed"
                        single_output = ""

                        try:
                            single_output = get_triples_from_local_model(model_mod, [img_path], prompt)
                            img_status = "success" if single_output else "failed"
                        except Exception as e:
                            img_error = str(e)
                            img_status = "failed"

                        img_latency = time.time() - t_img

                        # ADD: log after every single image
                        log_metrics(
                            scenario=scenario,
                            weather=weather,
                            loop=loop,
                            img_path=img_path,
                            prompt_text=prompt,
                            raw_output=single_output,
                            latency=img_latency,
                            status=img_status,
                            error=img_error
                        )

                        if single_output:
                            loop_raw_triples += "\n" + single_output
                            try:
                                if not single_output.startswith("@prefix"):
                                    single_output = prefixes + "\n" + single_output
                                temp.parse(data=single_output, format='turtle')
                            except Exception as parse_err:
                                print(f"      [!] Parse error on image: {parse_err}")
                    '''
                    # Process images one-by-one to avoid the Tensor Mismatch Error
                    for img_path in selected_images: #selected_images:
                        print(f"  -> Processing: {os.path.basename(img_path)}")
                        # Pass image as a single-item list
                        single_output = get_triples_from_local_model(model_mod, [img_path], prompt)
                        
                        if single_output:
                            loop_raw_triples += "\n" + single_output
                            # Parse into the temporary loop graph
                            try:
                                if not single_output.startswith("@prefix"):
                                    single_output = prefixes + "\n" + single_output
                                temp.parse(data=single_output, format='turtle')
                            except Exception as parse_err:
                                print(f"      [!] Parse error on image: {parse_err}")
                    '''
                    status = "success" # Only set to success if the loop finishes
                    
                    
                except Exception as e:
                    print(f"Inference failed: {e}")
                    status = "failed"

                latency = time.time() - t0
                cpu_time = time.process_time() - cpu0

                # Quality calculation
                try:
                    # Use the combined temp graph from all images in this batch
                    f1, halluc, compliance = tracker.calculate_quality(temp)
                except:
                    f1, halluc, compliance = 0.0, 100.0, 0.0

                # Now 'status' is guaranteed to exist
                model_stats.append({
                    "latency": latency,
                    "cpu_time": cpu_time,
                    "ram": ram,
                    "f1": f1,
                    "halluc": halluc,
                    "comp": compliance,
                    "status": status
                })

                # Confidence and Classifier scores
                avg_confidence_score = compute_avg_confidence_score(loop_raw_triples) or 0.0
                avg_classifier_score = compute_avg_classifier_score(selected_images) or 1.0
                adjusted_score = avg_confidence_score / avg_classifier_score

                # Add to the scenario's main graph
                main_graph = main_graph + temp
                print(f"Loop {loop} finished. Total triples in main graph: {len(main_graph)}")

                # metrics
                print(f"  -- Loop {loop} Metrics --")
                print(f"     Latency:        {latency:.2f}s")
                print(f"     CPU Time:       {cpu_time:.2f}s")
                print(f"     RAM:            {ram:.1f} MB")
                print(f"     F1:             {f1:.4f}")
                print(f"     Hallucination:  {halluc:.2f}%")
                print(f"     Compliance:     {compliance:.4f}")
                print(f"     Confidence:     {avg_confidence_score:.4f}")
                print(f"     Classifier:     {avg_classifier_score:.4f}")
                print(f"     Adjusted Score: {adjusted_score:.4f}")
                print(f"     Status:         {status}")
                print(f"  -------------------------")

                # Save logic...
                loop_output_path = os.path.join("output", model_name, scenario, weather, str(loop + 1))
                os.makedirs(loop_output_path, exist_ok=True)
                
                loop_output_file = os.path.join(loop_output_path, "vehicle_A_observations_loop.ttl")
                temp.serialize(destination=loop_output_file, format='turtle')
                
                main_output_file = os.path.join(loop_output_path, "vehicle_A_observations.ttl")
                main_graph.serialize(destination=main_output_file, format='turtle')

                loop += 1

            '''
                # Get the next 5 RGB images and first loop LIDAR images
            while (loop * model_limit) < len(rgb_images):

                print(f"Loop: {loop}, RGB images: {len(rgb_images)}")

                start_idx = loop * model_limit
                selected_images = rgb_images[start_idx : start_idx + model_limit]

                t0 = time.time()
                cpu0 = time.process_time()
                ram, vram = tracker.get_system_usage()


                try:
                    triples = get_triples_from_local_model(model_mod, selected_images, prompt)
                    status = "success"
                except Exception as e:
                    print(f"Inference failed: {e}")
                    triples = ""
                    status = "failed"

                latency = time.time() - t0
                cpu_time = time.process_time() - cpu0

                    # Add prefixes if not present
                if not triples.startswith("@prefix"):
                    triples = prefixes + "\n" + triples

                    
                temp = Graph()
                try:
                    if triples:
                        temp.parse(data=triples, format='turtle')
                    f1, halluc, compliance = tracker.calculate_quality(temp)
                except:
                    f1, halluc, compliance = 0.0, 100.0, 0.0

                    # Store metrics
                model_stats.append({
                    "latency": latency,
                    "cpu_time": cpu_time,
                    "ram": ram,
                    "f1": f1,
                    "halluc": halluc,
                    "comp": compliance,
                    "status": status
                })

                avg_confidence_score = compute_avg_confidence_score(triples)
                if avg_confidence_score is None:
                    avg_confidence_score = 0.0
                avg_classifier_score = compute_avg_classifier_score(selected_images)
                if avg_classifier_score is None:
                    avg_classifier_score = 1.0

                print(f"Average Confidence Score: {avg_confidence_score}, Classifier Score: {avg_classifier_score}")
                    # Calculate the total average score
                adjusted_score = avg_confidence_score / avg_classifier_score
                print("The adjusted score", adjusted_score)

                    # parse the triples
                temp = Graph()
                try:
                    temp.parse(data=triples, format='turtle')
                    main_graph = main_graph + temp
                except Exception as e:
                    print(f"Could not parse triples for {model_name}: {e}")

                    # Add to the main graph and exit the loop
                    # main_graph = main_graph + temp
                print(f"main graph has {len(main_graph)} triples.")

                loop_output_path = os.path.join("output", model_name, scenario, weather, str(loop + 1))
                os.makedirs(loop_output_path, exist_ok=True)

                    # Save the triples to a TTL file
                loop_output_file = os.path.join(loop_output_path, f"vehicle_A_observations_loop.ttl")
                temp.serialize(destination=loop_output_file, format='turtle')
                print(f"Loop graph with {len(temp)} triples saved to {loop_output_file}.")

                    # Save the main graph to a TTL file
                main_output_file = os.path.join(loop_output_path, "vehicle_A_observations.ttl")
                main_graph.serialize(destination=main_output_file, format='turtle')
                print(f"Main graph with {len(main_graph)} triples saved to {main_output_file}.")

                print(f"Confidence score: {adjusted_score}. Continuing to next loop.")
                loop += 1
                '''

    report = tracker.summarize_scenario(model_name, model_limit, model_stats)
    print(report)

    

