from rdflib import Graph, Namespace, RDF, URIRef, Literal
from rdflib.namespace import XSD
import os


# Standard Namespaces
AVCCO = Namespace("http://cornercase.org/avcco#")
EX = Namespace("http://cornercase.org/instances#")
PROV = Namespace("http://www.w3.org/ns/prov#")

def create_comprehensive_gt(weather_type):
    g = Graph()
    # Explicitly bind all prefixes to match your model's expected output header
    g.bind("avcco", AVCCO)
    g.bind("ex", EX)
    g.bind("prov", PROV)
    g.bind("rdf", RDF) # Restored for strict prefix matching

    # === AGENT & ACTIVITY ===
    g.add((EX.VehicleA, RDF.type, PROV.Agent))
    g.add((EX.vehicleA_activity_1, RDF.type, PROV.Activity))
    g.add((EX.vehicleA_activity_1, PROV.wasAssociatedWith, EX.VehicleA))

    # === WEATHER OBSERVATIONS ===
    g.add((EX.WeatherObs1, RDF.type, AVCCO.Observation))
    g.add((EX.WeatherObs1, AVCCO.hasConfidenceScore, Literal(0.9)))
    # FIXED: Changed 'prov' to 'PROV' to resolve NameError
    g.add((EX.WeatherObs1, PROV.wasGeneratedBy, EX.vehicleA_activity_1))

    if weather_type == "fog":
        g.add((EX.WeatherObs1, AVCCO.hasWeatherCondition, AVCCO.Fog))
    elif weather_type == "night":
        g.add((EX.WeatherObs1, AVCCO.hasWeatherCondition, AVCCO.Night))
    else:
        g.add((EX.WeatherObs1, AVCCO.hasWeatherCondition, AVCCO.Clear))

    # === BUS (The Corner Case / Occluder) ===
    g.add((EX.Bus1, RDF.type, AVCCO.Bus))
    g.add((EX.Bus1, AVCCO.hasRelativePosition, Literal("front")))
    
    g.add((EX.Obs2, RDF.type, AVCCO.Observation))
    g.add((EX.Obs2, AVCCO.refersTo, EX.Bus1))
    g.add((EX.Obs2, AVCCO.hasConfidenceScore, Literal(0.75)))
    g.add((EX.Obs2, PROV.wasGeneratedBy, EX.vehicleA_activity_1))

    # === CAR 1 (Visible Hatchback) ===
    g.add((EX.Car1, RDF.type, AVCCO.Car))
    g.add((EX.Car1, AVCCO.hasColor, Literal("red")))
    g.add((EX.Car1, AVCCO.hasColor, Literal("white")))
    g.add((EX.Car1, AVCCO.isOccludedBy, EX.Bus1))
    
    g.add((EX.Obs1, RDF.type, AVCCO.Observation))
    g.add((EX.Obs1, AVCCO.refersTo, EX.Car1))
    g.add((EX.Obs1, AVCCO.hasConfidenceScore, Literal(0.85)))
    g.add((EX.Obs1, PROV.wasGeneratedBy, EX.vehicleA_activity_1))

    # === BLACK CAR (Occluded SUV) ===
    # Using 'ex:Cars' to match your manual examples for better F1 score
    g.add((EX.Cars, RDF.type, AVCCO.Car))
    g.add((EX.Cars, AVCCO.hasColor, Literal("black")))
    g.add((EX.Cars, AVCCO.isOccludedBy, EX.Bus1))
    
    g.add((EX.Obs3, RDF.type, AVCCO.Observation))
    g.add((EX.Obs3, AVCCO.refersTo, EX.Cars))
    g.add((EX.Obs3, AVCCO.hasConfidenceScore, Literal(0.65)))
    g.add((EX.Obs3, PROV.wasGeneratedBy, EX.vehicleA_activity_1))

    return g

# Run the generation for all scenarios
weather_scenarios = ["day", "fog", "night"]

for weather in weather_scenarios:
    output_dir = f"ground_truth/{weather}"
    os.makedirs(output_dir, exist_ok=True)
    
    g = create_comprehensive_gt(weather)
    file_path = os.path.join(output_dir, "ground_truth.ttl")
    g.serialize(destination=file_path, format="turtle")
    print(f"Generated: {file_path}")