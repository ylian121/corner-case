import os
import glob
from rdflib import Graph, Namespace

# Ensure namespaces are recognized for comparison
AVCCO = Namespace("http://cornercase.org/avcco#")
EX = Namespace("http://cornercase.org/instances#")
PROV = Namespace("http://www.w3.org/ns/prov#")

ontology_path = "avcc_with_reasoning_no_shacl.ttl"
# Check if ontology exists to avoid crash
if os.path.exists(ontology_path):
    ontology = Graph().parse(ontology_path, format="turtle")
    ontology_predicates = set(ontology.predicates())
else:
    print(f"Warning: Ontology not found at {ontology_path}. Compliance will be 0.")
    ontology_predicates = set()

ttl_files = glob.glob("output/Qwen/**/*.ttl", recursive=True)
all_results = []

for ttl_path in sorted(ttl_files):
    parts = ttl_path.split(os.sep)
    
    # Adjusting based on your screenshot structure:
    # output/Qwen/bus_obscuring_car/day/1/file.ttl
    if len(parts) < 4: continue
    
    scenario = parts[2]
    weather  = parts[3]

    # Change this line if your GT is just in ground_truth/day/
    # If GT is in ground_truth/bus_obscuring_car/day/, keep scenario in path.
    gt_path = os.path.join("ground_truth", weather, "ground_truth.ttl")

    g = Graph()
    try:
        g.parse(ttl_path, format="turtle")
    except Exception as e:
        print(f"  [SKIP] {ttl_path}: {e}")
        continue

    gen_triples = set(g)
    total = len(gen_triples)

    # Compliance & Hallucination (Ontology check)
    if total == 0:
        compliance = 0.0
    else:
        valid = sum(1 for _, p, _ in gen_triples if p in ontology_predicates)
        compliance = round((valid / total) * 100, 2)

    # F1 against Ground Truth
    f1 = precision = recall = 0.0
    halluc_rate = 100.0 # Default if no GT found

    if os.path.exists(gt_path):
        gt = Graph().parse(gt_path, format="turtle")
        gt_triples = set(gt)
        
        # Exact triple match (Subject, Predicate, Object must be identical)
        tp = len(gen_triples & gt_triples)
        fp = len(gen_triples - gt_triples)
        fn = len(gt_triples - gen_triples)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0
        halluc_rate = round(fp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    else:
        print(f"  [MISSING GT] Could not find: {gt_path}")

    all_results.append({
        "file": ttl_path,
        "triples": total,
        "compliance": compliance,
        "hallucination": halluc_rate,
        "f1": f1,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    })


    print(f"{ttl_path}")
    print(f"  Triples:     {total}")
    print(f"  Compliance:  {compliance}%")
    print(f"  Halluc Rate: {halluc_rate}%")
    print(f"  Precision:   {round(precision,3)}")
    print(f"  Recall:      {round(recall,3)}")
    print(f"  F1:          {f1}")
    print()

if all_results:
    n = len(all_results)
    print("=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    print(f"  Files analyzed    : {n}")
    print(f"  Total triples     : {sum(r['triples'] for r in all_results)}")
    print(f"  Avg triples/file  : {round(sum(r['triples'] for r in all_results)/n, 2)}")
    print(f"  Avg Compliance    : {round(sum(r['compliance'] for r in all_results)/n, 2)}%")
    print(f"  Avg Halluc Rate   : {round(sum(r['hallucination'] for r in all_results)/n, 2)}%")
    print(f"  Avg Precision     : {round(sum(r['precision'] for r in all_results)/n, 3)}")
    print(f"  Avg Recall        : {round(sum(r['recall'] for r in all_results)/n, 3)}")
    print(f"  Avg F1            : {round(sum(r['f1'] for r in all_results)/n, 3)}")