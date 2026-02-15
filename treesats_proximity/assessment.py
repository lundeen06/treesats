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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich import box
from rich.layout import Layout
from rich.text import Text

# Initialize Rich console
console = Console()

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

    console.print(Panel(
        f"[cyan]Video:[/cyan] {video_path}\n"
        f"[cyan]Temp directory:[/cyan] {temp_dir}",
        title="EXTRACTING FRAMES",
        border_style="cyan"
    ))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    console.print(f"[yellow]Video info:[/yellow] {total_frames} frames @ {fps:.1f} fps ({duration:.1f}s)")
    console.print(f"[yellow]Extracting:[/yellow] Every {frame_interval} frames...")

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

    console.print(f"✓ [green]Extracted {extracted_count} frames[/green] from {total_frames} total frames")
    console.print(f"✓ [green]Sampling rate:[/green] {extracted_count/duration:.2f} frames/second\n")

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
        console.print(Panel.fit(
            "[bold red]ERROR: Cosmos Reason server is not running![/bold red]\n\n"
            "To start the server, run in a separate terminal:\n"
            "  [cyan]./start_cosmos.sh[/cyan]\n\n"
            "Or manually:\n"
            "  [cyan]docker run --runtime=nvidia --gpus all \\[/cyan]\n"
            "  [cyan]  --shm-size=32g -p 8000:8000 \\[/cyan]\n"
            "  [cyan]  -e NGC_API_KEY=$NGC_API_KEY \\[/cyan]\n"
            "  [cyan]  nvcr.io/nim/nvidia/cosmos-reason2-8b:latest[/cyan]\n\n"
            "Wait for it to finish loading, then run this script again.",
            title="SERVER NOT AVAILABLE",
            border_style="red"
        ))
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

                # Create Bayesian update table
                belief_shift = p_threat - prior_p
                shift_color = "red" if belief_shift > 0.05 else "green" if belief_shift < -0.05 else "yellow"
                uncertainty_reduction = prior_var - var_threat

                table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Prior", justify="right", style="blue")
                table.add_column("Observation", justify="right", style="yellow")
                table.add_column("Posterior", justify="right", style="bold green")

                table.add_row("p(threat)", f"{prior_p:.3f}", f"{obs_p_threat:.3f}", f"{p_threat:.3f}")
                table.add_row("σ² (uncertainty)", f"{prior_var:.3f}", f"{obs_var_threat:.3f}", f"{var_threat:.3f}")

                console.print(Panel(table, title="BAYESIAN BELIEF UPDATE", border_style="magenta"))
                console.print(f"  [bold {shift_color}]Belief shift:[/bold {shift_color}] {belief_shift:+.3f}")
                console.print(f"  [bold green]Uncertainty reduction:[/bold green] {uncertainty_reduction:.3f}\n")
            else:
                # No prior, use raw observation
                p_threat = obs_p_threat
                var_threat = obs_var_threat
                console.print(Panel(
                    f"[bold cyan]Initial observation:[/bold cyan]\n"
                    f"  p(threat) = [bold yellow]{p_threat:.3f}[/bold yellow]\n"
                    f"  σ² = [yellow]{var_threat:.3f}[/yellow]",
                    title="INITIAL ASSESSMENT",
                    border_style="cyan"
                ))

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

    console.print(Panel.fit(
        f"[bold cyan]Total frames:[/bold cyan] {len(image_paths)}\n"
        f"[bold cyan]Method:[/bold cyan] Contextual observations + Bayesian fusion",
        title="SEQUENTIAL VIDEO ASSESSMENT",
        border_style="bold cyan"
    ))

    for i, image_path in enumerate(image_paths):
        # Frame header
        console.rule(f"[bold blue]Frame {i+1}/{len(image_paths)}[/bold blue]", style="blue")
        console.print(f"[dim]{image_path}[/dim]\n")

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

        # Print frame summary with color-coded threat level
        threat = analysis['p_threat']
        if threat < 0.3:
            threat_color = "green"
            threat_label = "LOW"
        elif threat < 0.5:
            threat_color = "yellow"
            threat_label = "MEDIUM"
        elif threat < 0.7:
            threat_color = "orange1"
            threat_label = "ELEVATED"
        else:
            threat_color = "red"
            threat_label = "HIGH"

        summary_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Threat Level", f"[bold {threat_color}]{threat_label}[/bold {threat_color}]")
        summary_table.add_row("p(threat)", f"[{threat_color}]{threat:.3f}[/{threat_color}]")
        summary_table.add_row("Uncertainty (σ²)", f"{analysis['var_threat']:.3f}")
        summary_table.add_row("Confidence", f"{analysis.get('confidence', 0.0):.3f}")
        summary_table.add_row("Type", analysis.get('object_type', 'unknown'))

        console.print(Panel(summary_table, title=f"Frame {i+1} Summary", border_style=threat_color))

        # Display reasoning for this frame
        reasoning = analysis.get('notes', 'No reasoning provided')
        console.print(Panel(
            reasoning,
            title=f"Frame {i+1} Analysis",
            border_style=threat_color,
            padding=(1, 2)
        ))
        console.print()

    return assessments


def main(image_path: str, prior_belief: Dict = None):
    """
    Main space object assessment pipeline using Cosmos Reason VLM.

    Args:
        image_path: Path to space object image
        prior_belief: Optional prior assessment for Bayesian updating
    """
    console.print(f"\n[bold cyan]Analyzing space object:[/bold cyan] {image_path}")

    # Use Cosmos Reason VLM for complete analysis
    console.print("\n[bold cyan][1/2][/bold cyan] Analyzing with Cosmos Reason VLM...")
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

    # Threat level classification with colors
    if p_threat < 0.3:
        threat_level = "LOW"
        threat_color = "green"
    elif p_threat < 0.5:
        threat_level = "MEDIUM"
        threat_color = "yellow"
    elif p_threat < 0.7:
        threat_level = "ELEVATED"
        threat_color = "orange1"
    else:
        threat_level = "HIGH"
        threat_color = "red"

    # Main results table
    results_table = Table(title="SPACE OBJECT ASSESSMENT", title_style="bold white", box=box.DOUBLE_EDGE, show_header=False)
    results_table.add_column("Metric", style="bold cyan", width=20)
    results_table.add_column("Value", style="white")

    results_table.add_row("Image", os.path.basename(image_path))
    results_table.add_row("Object Type", f"[bold]{object_type.upper()}[/bold]")
    results_table.add_row("Subtype", object_subtype)
    results_table.add_row("", "")  # Spacer
    results_table.add_row("Threat Level", f"[bold {threat_color}]{threat_level}[/bold {threat_color}]")
    results_table.add_row("p(Threat)", f"[{threat_color}]{p_threat:.3f}[/{threat_color}]")
    results_table.add_row("Confidence", f"{confidence:.3f}")
    results_table.add_row("Uncertainty", f"{var_threat:.3f}")

    console.print()
    console.print(results_table)

    # Components table
    if components:
        console.print()
        comp_table = Table(title="Detected Components", box=box.ROUNDED, show_header=True)
        comp_table.add_column("Component", style="cyan")
        comp_table.add_column("Description", style="white")
        comp_table.add_column("Condition", justify="center")

        for component in components:
            name = component.get('name', 'unknown')
            desc = component.get('description', '')
            condition = component.get('condition', 'unknown')
            threat_lvl = component.get('threat_level', 'unknown')

            # Color code by threat level
            if threat_lvl == 'benign':
                cond_color = "green"
            elif threat_lvl == 'moderate':
                cond_color = "yellow"
            elif threat_lvl == 'elevated':
                cond_color = "orange1"
            else:
                cond_color = "white"

            comp_table.add_row(name, desc, f"[{cond_color}]{condition}[/{cond_color}]")

        console.print(comp_table)

    # Characteristics
    if characteristics:
        console.print()
        char_table = Table(title="Characteristics", box=box.ROUNDED, show_header=False)
        char_table.add_column("Property", style="cyan", width=20)
        char_table.add_column("Value", style="white")

        for key, value in characteristics.items():
            char_table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(char_table)

    # Threat Assessment Details
    if threat_assessment:
        console.print()
        threat_table = Table(title="Threat Assessment Details", box=box.ROUNDED, show_header=False)
        threat_table.add_column("Indicator", style="cyan", width=25)
        threat_table.add_column("Assessment", style="white")

        for key, value in threat_assessment.items():
            if isinstance(value, list):
                value_str = ', '.join(value) if value else 'none'
            else:
                value_str = str(value)
            threat_table.add_row(key.replace('_', ' ').title(), value_str)

        console.print(threat_table)

    # Reasoning panel
    console.print()
    console.print(Panel(
        notes,
        title="Analysis Reasoning",
        border_style=threat_color,
        padding=(1, 2)
    ))

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save JSON output
    console.print("\n[bold cyan][2/2][/bold cyan] Saving JSON results...")
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

    console.print(f"[green]✓ JSON saved to:[/green] {json_output_path}")
    console.print(f"\n[bold green]Analysis complete![/bold green] Results saved to {output_dir}/")

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

        # Final summary
        initial_threat = assessments[0]['p_threat']
        final_threat = assessments[-1]['p_threat']
        belief_shift = final_threat - initial_threat

        shift_color = "red" if belief_shift > 0.1 else "green" if belief_shift < -0.1 else "yellow"

        summary_table = Table(title="SEQUENTIAL ASSESSMENT COMPLETE", box=box.DOUBLE_EDGE, show_header=False)
        summary_table.add_column("Metric", style="bold cyan", width=20)
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Initial Threat", f"{initial_threat:.3f}")
        summary_table.add_row("Final Threat", f"[bold]{final_threat:.3f}[/bold]")
        summary_table.add_row("Belief Shift", f"[{shift_color}]{belief_shift:+.3f}[/{shift_color}]")
        summary_table.add_row("Frames Analyzed", f"{len(assessments)}")

        console.print()
        console.print(summary_table)
        console.print(f"\n[green]Results saved to:[/green] {sequence_output_path}")

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
            console.print(Panel.fit(
                f"[cyan]Input:[/cyan] {file_path}\n"
                f"[cyan]Target FPS:[/cyan] {target_fps}\n"
                f"[cyan]Max frames:[/cyan] {max_frames if max_frames else 'unlimited'}",
                title="VIDEO PROCESSING MODE",
                border_style="bold cyan"
            ))

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
                console.print(f"\n[dim]Cleaning up temporary frames: {temp_dir}[/dim]")
                shutil.rmtree(temp_dir)

            # Final video summary
            initial_threat = assessments[0]['p_threat']
            final_threat = assessments[-1]['p_threat']
            belief_shift = final_threat - initial_threat
            final_confidence = assessments[-1]['confidence']

            shift_color = "red" if belief_shift > 0.1 else "green" if belief_shift < -0.1 else "yellow"

            summary_table = Table(title="VIDEO ASSESSMENT COMPLETE", box=box.DOUBLE_EDGE, show_header=False)
            summary_table.add_column("Metric", style="bold cyan", width=20)
            summary_table.add_column("Value", justify="right")

            summary_table.add_row("Frames Analyzed", f"{len(frame_paths)}")
            summary_table.add_row("Initial Threat", f"{initial_threat:.3f}")
            summary_table.add_row("Final Threat", f"[bold]{final_threat:.3f}[/bold]")
            summary_table.add_row("Belief Shift", f"[{shift_color}]{belief_shift:+.3f}[/{shift_color}]")
            summary_table.add_row("Final Confidence", f"{final_confidence:.3f}")

            console.print()
            console.print(summary_table)
            console.print(f"\n[green]Results saved to:[/green] {video_output_path}")

        else:
            # Single image mode
            main(file_path)
