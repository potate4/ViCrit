#!/usr/bin/env python3
"""
Local ViCrit Setup Script
Helps you get started with running ViCrit locally on smaller models
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'datasets', 'PIL', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements_local.txt")
        return False
    return True

def download_dataset():
    """Download ViCrit dataset"""
    print("Downloading ViCrit dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("russwang/ViCrit-Bench", split="train")
        print(f"✅ Dataset downloaded successfully! ({len(dataset)} samples)")
        return True
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return False

def main():
    print("🚀 ViCrit Local Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        install = input("Install requirements? (y/n): ").lower().strip()
        if install == 'y':
            if not install_requirements():
                return
        else:
            print("Please install requirements manually and run again")
            return
    else:
        print("✅ All required packages are installed")
    
    # Download dataset
    download_dataset()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Choose a model from evaluation_local.sh")
    print("2. Run: ./evaluation_local.sh")
    print("3. Score results: python score_local.py")
    print("\nRecommended models for local use:")
    print("- llava-hf/llava-1.5-7b-hf (7B parameters, good balance)")
    print("- llava-hf/llava-1.5-13b-hf (13B parameters, better performance)")
    print("- Qwen/Qwen2.5-VL-7B-Instruct (7B parameters)")
    print("- Salesforce/instructblip-vicuna-7b (7B parameters, lightweight)")

if __name__ == "__main__":
    main() 