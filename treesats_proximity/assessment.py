#!/usr/bin/env python3
"""
Space Object Assessment using NVIDIA Cosmos Reason VLM
Identifies and assesses space objects (satellites, debris, fairings, etc.)
"""

import requests
import base64
from typing import Dict, Tuple
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables from .env file
load_dotenv()

# NVIDIA API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "not-used")
COSMOS_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
COSMOS_MODEL = "/model"

HEADERS = {
    "Content-Type": "application/json"
}


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def check_cosmos_server():
    """Check if Cosmos Reason server is running."""
    try:
        response = requests.get("http://127.0.0.1:8000/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False


def assess_with_cosmos(image_path: str) -> Tuple[float, Dict]:
    """
    Complete space object assessment using NVIDIA Cosmos Reason VLM.

    Args:
        image_path: Path to space object image

    Returns:
        Tuple of (threat_score, assessment_dict)
    """
    # Check if server is running
    if not check_cosmos_server():
        print("\n" + "!"*60)
        print("ERROR: Cosmos Reason server is not running!")
        print("!"*60)
        print("\nTo start the server, run in a separate terminal:")
        print("  ./start_cosmos.sh")
        print("\nOr manually:")
        print("  docker run --runtime=nvidia --gpus all \\")
        print("    --shm-size=32g -p 8000:8000 \\")
        print("    -e NGC_API_KEY=$NGC_API_KEY \\")
        print("    nvcr.io/nim/nvidia/cosmos-reason2-8b:latest")
        print("\nWait for it to finish loading, then run this script again.")
        print("!"*60 + "\n")
        raise ConnectionError("Cosmos Reason server not available")

    image_b64 = encode_image_to_base64(image_path)

    content = [
        {
            "type": "text",
            "text": (
                "You are analyzing a space object for situational awareness. "
                "This could be a satellite, debris, rocket fairing, upper stage, or unknown object.\n\n"
                "Provide a comprehensive analysis in ONLY valid JSON format (no markdown, no code blocks):\n\n"
                "{\n"
                '  "object_type": "satellite|debris|rocket_fairing|upper_stage|unknown",\n'
                '  "object_subtype": "active_satellite|derelict_satellite|fragment|spent_rocket_body|...",\n'
                '  "p_threat": 0.5,\n'
                '  "confidence": 0.85,\n'
                '  "components": [\n'
                '    {"name": "solar_panel", "description": "Extended solar arrays", "condition": "intact"},\n'
                '    {"name": "thruster", "description": "Visible propulsion system", "condition": "operational"}\n'
                '  ],\n'
                '  "characteristics": {\n'
                '    "size_estimate": "small|medium|large",\n'
                '    "orientation": "stable|tumbling|controlled",\n'
                '    "material_condition": "pristine|weathered|damaged|fragmented"\n'
                '  },\n'
                '  "threat_assessment": {\n'
                '    "maneuverability": "none|low|medium|high",\n'
                '    "proximity_risk": "low|medium|high",\n'
                '    "capability_indicators": ["list of observed capabilities"]\n'
                '  },\n'
                '  "reasoning": "Detailed analysis of why you classified it this way"\n'
                "}\n\n"
                "Field definitions:\n"
                "- object_type: primary classification of the space object\n"
                "- object_subtype: more specific classification\n"
                "- p_threat: threat/capability score 0.0-1.0 (0=passive/benign, 1=active/high-capability)\n"
                "- confidence: certainty in classification 0.0-1.0\n"
                "- components: list of visible parts/features (no bounding boxes needed)\n"
                "- characteristics: physical properties and state\n"
                "- threat_assessment: detailed capability analysis\n"
                "- reasoning: your analytical process and conclusions\n\n"
                "Return ONLY the JSON object, nothing else."
            )
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        }
    ]

    # Use local Cosmos Reason deployment
    payload = {
        "model": COSMOS_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "temperature": 0.1
    }

    response = requests.post(COSMOS_API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    result = response.json()

    if "choices" in result and len(result["choices"]) > 0:
        response_text = result["choices"][0].get("message", {}).get("content", "{}")
        print(f"\nCosmos Reason Response:\n{response_text}\n")

        # Remove markdown code blocks
        clean_text = re.sub(r'```json\s*|\s*```', '', response_text)
        clean_text = clean_text.strip()

        # Find JSON object
        json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)

        if json_match:
            data = json.loads(json_match.group())

            # Extract fields
            p_threat = float(data.get("p_threat", 0.5))
            confidence = float(data.get("confidence", 0.8))
            var_threat = 1.0 - confidence

            return (
                p_threat,
                {
                    "object_type": data.get("object_type", "unknown"),
                    "object_subtype": data.get("object_subtype", "unclassified"),
                    "p_threat": p_threat,
                    "var_threat": var_threat,
                    "confidence": confidence,
                    "components": data.get("components", []),
                    "characteristics": data.get("characteristics", {}),
                    "threat_assessment": data.get("threat_assessment", {}),
                    "notes": data.get("reasoning", "")
                }
            )

    raise ValueError("No valid response from Cosmos Reason")


def main(image_path: str):
    """
    Main space object assessment pipeline using Cosmos Reason VLM.

    Args:
        image_path: Path to space object image
    """
    print(f"Analyzing space object: {image_path}")

    # Use Cosmos Reason VLM for complete analysis
    print("\n[1/2] Analyzing with Cosmos Reason VLM...")
    threat_score, analysis = assess_with_cosmos(image_path)

    # Extract analysis data
    object_type = analysis.get('object_type', 'unknown')
    object_subtype = analysis.get('object_subtype', 'unclassified')
    p_threat = analysis.get('p_threat', threat_score)
    var_threat = analysis.get('var_threat', 0.2)
    confidence = analysis.get('confidence', 0.8)
    components = analysis.get('components', [])
    characteristics = analysis.get('characteristics', {})
    threat_assessment = analysis.get('threat_assessment', {})
    notes = analysis.get('notes', '')

    # Results
    print("\n" + "="*60)
    print("SPACE OBJECT ASSESSMENT RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"\nObject Type: {object_type.upper()}")
    print(f"Subtype: {object_subtype}")
    print(f"\nComponents/Features: {len(components)}")
    for component in components:
        name = component.get('name', 'unknown')
        desc = component.get('description', '')
        condition = component.get('condition', 'unknown')
        print(f"  - {name}: {desc} [{condition}]")

    if characteristics:
        print(f"\nCharacteristics:")
        for key, value in characteristics.items():
            print(f"  - {key}: {value}")

    if threat_assessment:
        print(f"\nThreat Assessment:")
        for key, value in threat_assessment.items():
            if isinstance(value, list):
                print(f"  - {key}: {', '.join(value)}")
            else:
                print(f"  - {key}: {value}")

    print(f"\nP(Threat): {p_threat:.3f}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Var(Threat): {var_threat:.3f}")

    # Threat level classification
    if p_threat < 0.3:
        threat_level = "LOW"
    elif p_threat < 0.7:
        threat_level = "MEDIUM"
    else:
        threat_level = "HIGH"

    print(f"\nThreat Level: {threat_level}")
    print(f"\nReasoning:\n{notes}")
    print("="*60)

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save JSON output
    print("\n[2/2] Saving JSON results...")
    json_output_path = os.path.join(output_dir, f"{base_name}_assessment.json")

    output_data = {
        "image": os.path.abspath(image_path),
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "object_type": object_type,
        "object_subtype": object_subtype,
        "p_threat": p_threat,
        "var_threat": var_threat,
        "confidence": confidence,
        "threat_level": threat_level,
        "components": components,
        "characteristics": characteristics,
        "threat_assessment": threat_assessment,
        "reasoning": notes
    }

    with open(json_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ JSON saved to: {json_output_path}")
    print(f"\nAnalysis complete! Results saved to {output_dir}/")

    return p_threat


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python assessment.py <image_path>")
        print("\nExample: python assessment.py data/sat1.jpg")
        print("\nNote: Make sure Cosmos Reason is running locally:")
        print("  docker run --runtime=nvidia --gpus all \\")
        print("    --shm-size=32g -p 8000:8000 \\")
        print("    -e NGC_API_KEY=$NGC_API_KEY \\")
        print("    nvcr.io/nim/nvidia/cosmos-reason2-8b:latest")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    main(image_path)
