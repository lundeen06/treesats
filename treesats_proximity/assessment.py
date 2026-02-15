#!/usr/bin/env python3
"""
Space Object Assessment using NVIDIA Cosmos Reason VLM
Identifies and assesses space objects (satellites, debris, fairings, etc.)
Supports single images, image sequences, and video files (mp4, avi, mov)
"""

import requests
import base64
from typing import Dict, Tuple, List
import os
from dotenv import load_dotenv
import json
import re
import tempfile
import shutil

# Load environment variables from .env file
load_dotenv()

# NVIDIA API Configuration
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


def extract_frames_from_video(video_path: str, frame_interval: int = 30, max_frames: int = None) -> List[str]:
    """
    Extract frames from a video file for sequential analysis.

    Args:
        video_path: Path to video file (mp4, avi, mov, etc.)
        frame_interval: Extract every Nth frame (default: 30 = ~1 fps for 30fps video)
        max_frames: Maximum number of frames to extract (None = all)

    Returns:
        List of paths to extracted frame images
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for video processing. Install with:\n"
            "  pip install opencv-python"
        )

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="treesats_frames_")
    print(f"\nExtracting frames from video: {video_path}")
    print(f"Temporary frames directory: {temp_dir}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video info: {total_frames} frames @ {fps:.1f} fps ({duration:.1f}s)")
    print(f"Extracting every {frame_interval} frames...")

    frame_paths = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(temp_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(frame_filename)
            extracted_count += 1

            if max_frames and extracted_count >= max_frames:
                print(f"Reached max_frames limit ({max_frames})")
                break

        frame_count += 1

    cap.release()

    print(f"Extracted {extracted_count} frames from {total_frames} total frames")
    print(f"Effective sampling rate: {extracted_count/duration:.2f} frames/second\n")

    return frame_paths


def bayesian_update(prior_p: float, prior_var: float, obs_p: float, obs_var: float) -> Tuple[float, float]:
    """
    Bayesian update of threat belief using Kalman-like fusion.

    Args:
        prior_p: Prior mean threat probability
        prior_var: Prior variance/uncertainty
        obs_p: Observed threat probability from current frame
        obs_var: Observation variance/uncertainty

    Returns:
        Tuple of (posterior_p, posterior_var)
    """
    # Kalman gain: how much to trust new observation vs prior
    kalman_gain = prior_var / (prior_var + obs_var)

    # Update mean: weighted combination of prior and observation
    posterior_p = prior_p + kalman_gain * (obs_p - prior_p)

    # Update variance: uncertainty decreases as we gather more evidence
    posterior_var = (1 - kalman_gain) * prior_var

    # Ensure values stay in valid range
    posterior_p = max(0.0, min(1.0, posterior_p))
    posterior_var = max(0.01, min(0.5, posterior_var))  # Keep some minimum uncertainty

    return posterior_p, posterior_var


def assess_with_cosmos(image_path: str, prior_belief: Dict = None) -> Tuple[float, Dict]:
    """
    Complete space object assessment using NVIDIA Cosmos Reason VLM with Bayesian belief updating.

    Args:
        image_path: Path to space object image
        prior_belief: Optional prior assessment from previous frame/observation
                     Format: {"p_threat": float, "var_threat": float, "notes": str}

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

    # Build prompt with qualitative prior context (NO numerical p_threat to avoid anchoring)
    prior_context = ""
    if prior_belief:
        prior_notes = prior_belief.get('notes', 'None')
        prior_context = (
            f"\n\nCONTEXT FROM PREVIOUS FRAME:\n"
            f"Previous observations: {prior_notes}\n\n"
            f"Use this context to track changes, but make your OWN independent threat assessment "
            f"based on what you see in THIS frame. Do NOT anchor on any previous numerical estimates. "
            f"Note what has changed, what has stayed the same, or what new features are now visible."
        )

    content = [
        {
            "type": "text",
            "text": (
                "You are an expert space situational awareness analyst evaluating potential threats. "
                "Assess this space object carefully, considering both passive characteristics and active threat indicators.\n\n"
                f"{prior_context}\n\n"
                "CRITICAL - AVOID HALLUCINATION:\n"
                "• ONLY report components you can CLEARLY see with high confidence\n"
                "• Do NOT infer or imagine components from ambiguous pixels or shadows\n"
                "• If you cannot clearly identify something, mark it as 'unknown' or 'unclear'\n"
                "• When in doubt, describe what you see literally (e.g., 'extended appendage') not what you think it might be\n"
                "• Absence of evidence is NOT evidence of absence - but don't fabricate evidence\n\n"
                "UNCERTAINTY & PRECISION GUIDANCE:\n"
                "• Single images show limited viewing angles - many components may be hidden\n"
                "• Be appropriately uncertain given observational limitations\n"
                "• Provide p_threat to 3 decimal places (e.g., 0.247, 0.683, NOT 0.2 or 0.7)\n"
                "• Provide confidence to 3 decimal places (e.g., 0.523, NOT 0.5)\n"
                "• Avoid rounding to convenient intervals - small differences matter\n"
                "• Real observations have NATURAL VARIANCE - each frame will show slightly different lighting, angles, aspects\n"
                "• DO NOT give the same probability twice - each observation is unique with measurement noise\n"
                "• Vary your estimates frame-to-frame based on what's visible/obscured in THIS specific view\n\n"
                "THREAT CALIBRATION GUIDELINES:\n"
                "• LOW RISK (0.0-0.3): Passive satellites with scientific/civilian purpose\n"
                "  - Examples: Cassini-class spacecraft, weather satellites, telescopes\n"
                "  - Indicators: Large solar panels, science instruments, stable orientation, no weaponization\n"
                "  - These are cooperative assets with predictable trajectories\n\n"
                "• MEDIUM RISK (0.3-0.5): Dual-use or ambiguous objects\n"
                "  - Examples: Communication satellites with maneuvering capability, objects with unclear purpose\n"
                "  - Indicators: Some maneuverability, close proximity approaches, obscured components\n"
                "  - Requires monitoring but not immediate concern\n\n"
                "• ELEVATED RISK (0.5-0.7): Objects with concerning characteristics\n"
                "  - Examples: Highly maneuverable objects, satellites with grappling/manipulation systems\n"
                "  - Indicators: Robotic arms, frequent trajectory changes, proximity operations without coordination\n\n"
                "• HIGH RISK (0.7-1.0): Objects with clear threat indicators or weaponization potential\n"
                "  - Examples: Objects resembling weapons platforms, aggressive proximity behavior, military designs\n"
                "  - Indicators: Weapon-like appendages, kinetic kill vehicle characteristics, aggressive maneuvers\n"
                "  - Includes any object clearly designed for offensive/defensive space operations\n\n"
                "IF YOU CLEARLY OBSERVE THESE, THEY ARE THREAT INDICATORS:\n"
                "1. **Maneuverability**: IF you see visible propulsion systems or attitude control thrusters\n"
                "2. **Weaponization**: IF you see articulated arms, manipulators, or weapon-like appendages (DO NOT assume these exist)\n"
                "3. **Design Purpose**: Assess whether visible design suggests civilian/scientific vs military purpose\n"
                "Note: Most satellites will NOT have weapons or robotic arms - the default assumption is benign unless clearly visible.\n\n"
                "Provide analysis in ONLY valid JSON format (no markdown, no code blocks):\n\n"
                "{\n"
                '  "object_type": "satellite|debris|rocket_fairing|upper_stage|unknown|weapon_system",\n'
                '  "object_subtype": "science_probe|comm_sat|military_sat|kinetic_kill_vehicle|robotic_servicing|...",\n'
                '  "p_threat": 0.5,\n'
                '  "confidence": 0.85,\n'
                '  "components": [\n'
                '    {"name": "solar_panel", "description": "Extended solar arrays for power", "threat_level": "benign"},\n'
                '    {"name": "robotic_arm", "description": "Articulated manipulator", "threat_level": "elevated"},\n'
                '    {"name": "thruster", "description": "High-thrust propulsion system", "threat_level": "moderate"}\n'
                '  ],\n'
                '  "characteristics": {\n'
                '    "size_estimate": "small|medium|large",\n'
                '    "orientation": "stable|tumbling|controlled|aggressive",\n'
                '    "material_condition": "pristine|weathered|damaged|fragmented",\n'
                '    "design_signature": "civilian|dual_use|military|weapon_platform"\n'
                '  },\n'
                '  "threat_assessment": {\n'
                '    "maneuverability": "none|low|medium|high",\n'
                '    "weaponization_indicators": ["list specific weapon-like features or \'none\'"],\n'
                '    "proximity_risk": "low|medium|high|critical",\n'
                '    "behavioral_anomalies": ["list unusual behaviors or \'none\'"],\n'
                '    "capability_indicators": ["list all observed capabilities"]\n'
                '  },\n'
                '  "reasoning": "Detailed analysis explaining your p_threat score. Reference specific visual evidence. If updating from prior, explain what new evidence changed or confirmed your assessment."\n'
                "}\n\n"
                "CRITICAL: Calibrate p_threat based on ACTUAL THREAT INDICATORS, not just capability:\n"
                "- Science satellites (Cassini-like): 0.1-0.2 even if maneuverable\n"
                "- Ambiguous objects (Tango-like): 0.3-0.5 based on uncertainty\n"
                "- Objects with robotic arms/manipulators: 0.6-0.8\n"
                "- Weapon-like designs (tie fighter, kinetic kill vehicles): 0.8-1.0\n\n"
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
        "temperature": 0.7  # Higher temperature for more variance/noise between frames
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

            # Extract raw observation from VLM
            obs_p_threat = float(data.get("p_threat", 0.5))
            confidence = float(data.get("confidence", 0.8))

            # Boost initial uncertainty - single images don't show all angles
            # Start with higher variance (lower confidence) since we have limited perspective
            obs_var_threat = 1.0 - confidence
            if not prior_belief:
                # For first observation, increase uncertainty by 50%
                obs_var_threat = min(0.5, obs_var_threat * 1.5)
                confidence = 1.0 - obs_var_threat

            # Apply Bayesian update if we have a prior
            if prior_belief:
                prior_p = prior_belief.get("p_threat", 0.5)
                prior_var = prior_belief.get("var_threat", 0.3)

                # Fuse prior with current observation
                p_threat, var_threat = bayesian_update(
                    prior_p=prior_p,
                    prior_var=prior_var,
                    obs_p=obs_p_threat,
                    obs_var=obs_var_threat
                )

                print(f"\n[Bayesian Update]")
                print(f"  Prior: p={prior_p:.3f}, var={prior_var:.3f}")
                print(f"  Observation: p={obs_p_threat:.3f}, var={obs_var_threat:.3f}")
                print(f"  Posterior: p={p_threat:.3f}, var={var_threat:.3f}")
                print(f"  Belief shift: {p_threat - prior_p:+.3f}")
            else:
                # No prior, use raw observation
                p_threat = obs_p_threat
                var_threat = obs_var_threat
                print(f"\n[Initial Assessment]")
                print(f"  Observation: p={p_threat:.3f}, var={var_threat:.3f}")

            # Update confidence based on posterior uncertainty
            confidence = 1.0 - var_threat

            return (
                p_threat,
                {
                    "object_type": data.get("object_type", "unknown"),
                    "object_subtype": data.get("object_subtype", "unclassified"),
                    "p_threat": p_threat,
                    "var_threat": var_threat,
                    "confidence": confidence,
                    "obs_p_threat": obs_p_threat,  # Store raw observation
                    "obs_var_threat": obs_var_threat,
                    "components": data.get("components", []),
                    "characteristics": data.get("characteristics", {}),
                    "threat_assessment": data.get("threat_assessment", {}),
                    "notes": data.get("reasoning", "")
                }
            )

    raise ValueError("No valid response from Cosmos Reason")


def assess_sequence(image_paths: list) -> list:
    """
    Sequentially assess multiple images (video frames) with Bayesian belief chaining.

    Args:
        image_paths: List of paths to images/frames in temporal order

    Returns:
        List of assessment dictionaries, one per frame
    """
    assessments = []
    prior_belief = None

    print(f"\n{'='*60}")
    print(f"SEQUENTIAL ASSESSMENT: {len(image_paths)} frames")
    print(f"{'='*60}\n")

    for i, image_path in enumerate(image_paths):
        print(f"\n--- Frame {i+1}/{len(image_paths)}: {image_path} ---")

        # Assess current frame with prior from previous frame
        threat_score, analysis = assess_with_cosmos(image_path, prior_belief=prior_belief)

        # Add frame metadata
        analysis['frame_number'] = i + 1
        analysis['image_path'] = image_path
        assessments.append(analysis)

        # Update prior for next frame
        prior_belief = {
            'p_threat': analysis['p_threat'],
            'var_threat': analysis['var_threat'],
            'notes': analysis.get('notes', '')
        }

        # Print frame summary
        print(f"\nFrame {i+1} Summary:")
        print(f"  Threat: {analysis['p_threat']:.3f} ± {analysis['var_threat']:.3f}")
        print(f"  Type: {analysis.get('object_type', 'unknown')}")
        print(f"  Confidence: {analysis.get('confidence', 0.0):.3f}")

    return assessments


def main(image_path: str, prior_belief: Dict = None):
    """
    Main space object assessment pipeline using Cosmos Reason VLM.

    Args:
        image_path: Path to space object image
        prior_belief: Optional prior assessment for Bayesian updating
    """
    print(f"Analyzing space object: {image_path}")

    # Use Cosmos Reason VLM for complete analysis
    print("\n[1/2] Analyzing with Cosmos Reason VLM...")
    threat_score, analysis = assess_with_cosmos(image_path, prior_belief=prior_belief)

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
        "obs_p_threat": analysis.get("obs_p_threat", p_threat),  # Raw observation
        "obs_var_threat": analysis.get("obs_var_threat", var_threat),
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
    import glob as glob_module

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python assessment.py <image_path>")
        print("  Video file:    python assessment.py <video_path> [--fps N] [--max-frames N]")
        print("  Sequential:    python assessment.py --sequence <pattern>")
        print("\nExamples:")
        print("  python assessment.py data/sat1.jpg")
        print("  python assessment.py data/proximity_event.mp4")
        print("  python assessment.py data/proximity_event.mp4 --fps 1 --max-frames 20")
        print("  python assessment.py --sequence 'data/frames/*.jpg'")
        print("\nOptions:")
        print("  --fps N          Extract N frames per second from video (default: 1)")
        print("  --max-frames N   Limit to first N frames from video")
        print("\nNote: Make sure Cosmos Reason is running locally:")
        print("  ./start_cosmos.sh")
        print("  # or manually:")
        print("  docker run --runtime=nvidia --gpus all \\")
        print("    --shm-size=32g -p 8000:8000 \\")
        print("    -e NGC_API_KEY=$NGC_API_KEY \\")
        print("    nvcr.io/nim/nvidia/cosmos-reason2-8b:latest")
        sys.exit(1)

    # Check for sequential mode
    if sys.argv[1] == "--sequence":
        if len(sys.argv) < 3:
            print("Error: --sequence requires at least one image path or pattern")
            sys.exit(1)

        # Collect all image paths
        image_paths = []
        for arg in sys.argv[2:]:
            # Try glob pattern expansion
            matches = glob_module.glob(arg)
            if matches:
                image_paths.extend(sorted(matches))
            elif os.path.exists(arg):
                image_paths.append(arg)
            else:
                print(f"Warning: No files found for pattern: {arg}")

        if not image_paths:
            print("Error: No valid image files found")
            sys.exit(1)

        # Verify all files exist
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Error: Image file not found: {path}")
                sys.exit(1)

        print(f"Processing {len(image_paths)} frames sequentially...")

        # Run sequential assessment
        assessments = assess_sequence(image_paths)

        # Save combined results
        output_dir = os.path.join(os.path.dirname(os.path.abspath(image_paths[0])), "..", "output")
        os.makedirs(output_dir, exist_ok=True)

        sequence_output_path = os.path.join(output_dir, "sequence_assessment.json")
        sequence_data = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "num_frames": len(image_paths),
            "frames": assessments,
            "final_threat": assessments[-1]['p_threat'] if assessments else 0.0,
            "final_confidence": assessments[-1]['confidence'] if assessments else 0.0,
            "threat_evolution": [a['p_threat'] for a in assessments]
        }

        with open(sequence_output_path, 'w') as f:
            json.dump(sequence_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"SEQUENTIAL ASSESSMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Initial threat: {assessments[0]['p_threat']:.3f}")
        print(f"Final threat: {assessments[-1]['p_threat']:.3f}")
        print(f"Belief shift: {assessments[-1]['p_threat'] - assessments[0]['p_threat']:+.3f}")
        print(f"\nResults saved to: {sequence_output_path}")

    else:
        # Single file mode (image or video)
        file_path = sys.argv[1]

        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        # Parse optional arguments
        target_fps = 1.0  # Default: 1 frame per second
        max_frames = None

        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--fps" and i + 1 < len(sys.argv):
                target_fps = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--max-frames" and i + 1 < len(sys.argv):
                max_frames = int(sys.argv[i + 1])
                i += 2
            else:
                print(f"Warning: Unknown argument: {sys.argv[i]}")
                i += 1

        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in video_extensions:
            # Video processing mode
            print(f"\n{'='*60}")
            print(f"VIDEO PROCESSING MODE")
            print(f"{'='*60}")
            print(f"Input: {file_path}")
            print(f"Target FPS: {target_fps}")
            print(f"Max frames: {max_frames if max_frames else 'unlimited'}")

            # Extract frames
            try:
                import cv2
                cap = cv2.VideoCapture(file_path)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                # Calculate frame interval
                frame_interval = max(1, int(video_fps / target_fps))

            except ImportError:
                print("\nERROR: OpenCV required for video processing")
                print("Install with: pip install opencv-python")
                sys.exit(1)

            frame_paths = extract_frames_from_video(
                file_path,
                frame_interval=frame_interval,
                max_frames=max_frames
            )

            if not frame_paths:
                print("Error: No frames extracted from video")
                sys.exit(1)

            # Run sequential assessment on extracted frames
            assessments = assess_sequence(frame_paths)

            # Save combined results
            output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), "..", "output")
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            video_output_path = os.path.join(output_dir, f"{base_name}_video_assessment.json")

            video_data = {
                "video_file": os.path.abspath(file_path),
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "num_frames_analyzed": len(frame_paths),
                "target_fps": target_fps,
                "frames": assessments,
                "initial_threat": assessments[0]['p_threat'] if assessments else 0.0,
                "final_threat": assessments[-1]['p_threat'] if assessments else 0.0,
                "final_confidence": assessments[-1]['confidence'] if assessments else 0.0,
                "threat_evolution": [a['p_threat'] for a in assessments],
                "confidence_evolution": [a['confidence'] for a in assessments]
            }

            with open(video_output_path, 'w') as f:
                json.dump(video_data, f, indent=2)

            # Clean up temporary frames
            if frame_paths:
                temp_dir = os.path.dirname(frame_paths[0])
                print(f"\nCleaning up temporary frames: {temp_dir}")
                shutil.rmtree(temp_dir)

            print(f"\n{'='*60}")
            print(f"VIDEO ASSESSMENT COMPLETE")
            print(f"{'='*60}")
            print(f"Frames analyzed: {len(frame_paths)}")
            print(f"Initial threat: {assessments[0]['p_threat']:.3f}")
            print(f"Final threat: {assessments[-1]['p_threat']:.3f}")
            print(f"Belief shift: {assessments[-1]['p_threat'] - assessments[0]['p_threat']:+.3f}")
            print(f"Final confidence: {assessments[-1]['confidence']:.3f}")
            print(f"\nResults saved to: {video_output_path}")

        else:
            # Single image mode
            main(file_path)
