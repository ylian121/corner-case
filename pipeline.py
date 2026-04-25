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
