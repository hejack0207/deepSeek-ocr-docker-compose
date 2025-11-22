#!/usr/bin/env python3
"""
Test script for DeepSeek OCR API
"""

import requests
import sys
import time
from pathlib import Path


API_URL = "http://localhost:8000"


def test_health():
    """Check API health"""
    print("🏥 Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        data = response.json()
        
        if data.get("status") == "healthy":
            print("✅ API is healthy")
            print(f"   - Model loaded: {data.get('model_loaded')}")
            print(f"   - Device: {data.get('device')}")
            print(f"   - CUDA available: {data.get('cuda_available')}")
            return True
        else:
            print("❌ API is not healthy")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False


def test_modes():
    """Get available modes"""
    print("\n📋 Getting available modes...")
    try:
        response = requests.get(f"{API_URL}/api/modes", timeout=5)
        modes = response.json()
        
        print("✅ Available modes:")
        for mode, info in modes.get("modes", {}).items():
            print(f"\n   {mode}:")
            print(f"   - {info['description']}")
            print(f"   - Speed: {info['speed']}")
            print(f"   - Use case: {info['use_case']}")
        return True
    except Exception as e:
        print(f"❌ Error getting modes: {e}")
        return False


def test_ocr(image_path, mode="markdown"):
    """Test OCR with an image"""
    print(f"\n\ud83d\udc50  Testing OCR with image: {image_path}")
    print(f"   Mode: {mode}")
    
    if not Path(image_path).exists():
        print(f"\u274c File not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'mode': mode}
            
            print("   \u23f3 Processing...")
            start = time.time()
            
            response = requests.post(
                f"{API_URL}/api/ocr",
                files=files,
                data=data,
                timeout=300  # Maximum 5 minutes
            )
            
            elapsed = time.time() - start
            
            if response.ok:
                result = response.json()
                print(f"\u2705 OCR completed in {result['processing_time']}s")
                print(f"   - Image size: {result['image_size']}")
                print(f"   - Mode used: {result['mode']}")
                print(f"\n   \ud83d\udcc4 Extracted text (first 200 chars):")
                print(f"   {result['text'][:200]}...")
                return True
            else:
                print(f"\u274c OCR error: {response.status_code}")
                print(f"   {response.text}")
                return False
                
    except Exception as e:
        print(f"\u274c Error processing OCR: {e}")
        return False


def run_tests(image_path=None):
    """Run all tests"""
    print("=" * 60)
    print("🧪 Starting DeepSeek OCR API tests")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health()
    
    if not results['health']:
        print("\n❌ API is not available. Make sure it's running:")
        print("   docker-compose up -d")
        return False
    
    # Test 2: Modes
    results['modes'] = test_modes()
    
    # Test 3: OCR (if image is provided)
    if image_path:
        results['ocr'] = test_ocr(image_path)
    else:
        print("\n⚠️  No image provided for OCR test")
        print("   Usage: python test_api.py <image_path>")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test summary:")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")
    
    return all_passed


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_tests(image_path)
    sys.exit(0 if success else 1)
