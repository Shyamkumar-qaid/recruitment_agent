#!/usr/bin/env python
"""
Dependency Resolution Script

This script helps resolve dependency conflicts by:
1. Checking the current environment for conflicts
2. Updating packages to compatible versions
3. Generating a clean requirements file

Usage:
    python update_dependencies.py
"""

import subprocess
import sys
import re
import os
from typing import List, Dict, Tuple

def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages and their versions."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True,
        text=True,
        check=True
    )
    
    import json
    packages = json.loads(result.stdout)
    return {pkg["name"].lower(): pkg["version"] for pkg in packages}

def check_for_conflicts() -> List[str]:
    """Check for dependency conflicts in the current environment."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True
    )
    
    conflicts = []
    if result.returncode != 0:
        # Parse the output to extract conflict information
        for line in result.stderr.splitlines() + result.stdout.splitlines():
            if "dependency conflict" in line.lower() or "incompatible" in line.lower():
                conflicts.append(line.strip())
    
    return conflicts

def parse_requirements(file_path: str) -> List[Tuple[str, str]]:
    """Parse requirements file into package name and version specifier pairs."""
    if not os.path.exists(file_path):
        return []
    
    requirements = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Handle inline comments
            if '#' in line:
                line = line.split('#')[0].strip()
            
            # Extract package name and version specifier
            match = re.match(r'^([a-zA-Z0-9_\-\.]+)(.*)$', line)
            if match:
                package_name = match.group(1).lower()
                version_spec = match.group(2).strip()
                requirements.append((package_name, version_spec))
    
    return requirements

def resolve_conflicts(conflicts: List[str], requirements: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Attempt to resolve conflicts by updating version specifiers."""
    # Extract package names and version constraints from conflicts
    conflict_info = {}
    for conflict in conflicts:
        # Example: "pyasn1-modules 0.4.2 requires pyasn1<0.7.0,>=0.6.1, but you have pyasn1 0.4.8"
        match = re.search(r'([a-zA-Z0-9_\-\.]+) .* requires ([a-zA-Z0-9_\-\.]+)([<>=].+?), but', conflict)
        if match:
            requiring_pkg = match.group(1).lower()
            required_pkg = match.group(2).lower()
            version_constraint = match.group(3)
            
            if required_pkg not in conflict_info:
                conflict_info[required_pkg] = []
            conflict_info[required_pkg].append((requiring_pkg, version_constraint))
    
    # Update requirements based on conflicts
    updated_requirements = []
    for pkg_name, version_spec in requirements:
        if pkg_name in conflict_info:
            # Combine all version constraints
            constraints = [c for _, c in conflict_info[pkg_name]]
            new_version_spec = ','.join(constraints)
            print(f"Updating {pkg_name} version constraint to: {new_version_spec}")
            updated_requirements.append((pkg_name, new_version_spec))
        else:
            updated_requirements.append((pkg_name, version_spec))
    
    # Add any missing packages from conflicts
    for pkg_name in conflict_info:
        if not any(name == pkg_name for name, _ in updated_requirements):
            constraints = [c for _, c in conflict_info[pkg_name]]
            new_version_spec = ','.join(constraints)
            print(f"Adding {pkg_name}{new_version_spec} to requirements")
            updated_requirements.append((pkg_name, new_version_spec))
    
    return updated_requirements

def update_packages(requirements: List[Tuple[str, str]]) -> None:
    """Update packages to the specified versions."""
    for pkg_name, version_spec in requirements:
        if version_spec:
            pkg_spec = f"{pkg_name}{version_spec}"
        else:
            pkg_spec = pkg_name
        
        print(f"Installing {pkg_spec}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", pkg_spec],
            check=True
        )

def generate_requirements_file(requirements: List[Tuple[str, str]], output_file: str) -> None:
    """Generate a new requirements file with updated version specifiers."""
    with open(output_file, 'w') as f:
        for pkg_name, version_spec in requirements:
            if version_spec:
                f.write(f"{pkg_name}{version_spec}\n")
            else:
                f.write(f"{pkg_name}\n")

def main():
    """Main function to resolve dependency conflicts."""
    print("Checking for dependency conflicts...")
    conflicts = check_for_conflicts()
    
    if not conflicts:
        print("No conflicts found!")
        return
    
    print(f"Found {len(conflicts)} conflicts:")
    for conflict in conflicts:
        print(f"  - {conflict}")
    
    print("\nParsing requirements.txt...")
    requirements = parse_requirements("requirements.txt")
    
    print("\nAttempting to resolve conflicts...")
    updated_requirements = resolve_conflicts(conflicts, requirements)
    
    print("\nUpdating packages...")
    update_packages(updated_requirements)
    
    print("\nGenerating updated requirements file...")
    generate_requirements_file(updated_requirements, "requirements.resolved.txt")
    
    print("\nChecking for remaining conflicts...")
    remaining_conflicts = check_for_conflicts()
    
    if not remaining_conflicts:
        print("All conflicts resolved successfully!")
        print("Updated requirements saved to requirements.resolved.txt")
    else:
        print(f"There are still {len(remaining_conflicts)} conflicts:")
        for conflict in remaining_conflicts:
            print(f"  - {conflict}")
        print("Manual intervention may be required.")

if __name__ == "__main__":
    main()