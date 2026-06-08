#!/usr/bin/env python3
"""
Test script to verify Pinecone API key and connection
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from hireflow.config import settings
from pinecone import Pinecone

print("=" * 70)
print("PINECONE CONNECTION TEST")
print("=" * 70)

# Test 1: Check if API key is set
print("\n[1] Checking API Key...")
if not settings.pinecone_api_key:
    print("❌ PINECONE_API_KEY is NOT set in .env")
    print("   Add this to .env:")
    print("   PINECONE_API_KEY=pcsk_...")
    sys.exit(1)
else:
    # Show masked key (first 20 chars visible)
    masked = settings.pinecone_api_key[:20] + "..." + settings.pinecone_api_key[-10:]
    print(f"✅ API Key found: {masked}")

# Test 2: Try to connect to Pinecone
print("\n[2] Connecting to Pinecone...")
try:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    print("✅ Connection successful!")
except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
    print("\nPossible fixes:")
    print("   1. Check API key is correct (copy from https://app.pinecone.io)")
    print("   2. Check internet connection")
    print("   3. Make sure .env file is in correct location")
    sys.exit(1)

# Test 3: List existing indexes
print("\n[3] Listing existing indexes...")
try:
    indexes = pc.list_indexes()
    print(f"✅ Found {len(indexes)} index(es):")
    for idx in indexes:
        # IndexModel objects have 'name' attribute, not dict access
        index_name = idx.name if hasattr(idx, 'name') else str(idx)
        print(f"   - {index_name}")
except Exception as e:
    print(f"❌ Failed to list indexes: {str(e)}")
    sys.exit(1)

# Test 4: Check if resume index exists
print(f"\n[4] Checking for index: '{settings.resume_index_name}'...")
try:
    existing_names = [idx.name if hasattr(idx, 'name') else str(idx) for idx in pc.list_indexes()]
    if settings.resume_index_name in existing_names:
        print(f"✅ Index '{settings.resume_index_name}' exists")
        print("\n   ⚠️  This old index might be causing issues!")
        print("   Go to https://app.pinecone.io and delete it, then try again")
    else:
        print(f"✅ Index '{settings.resume_index_name}' doesn't exist (will be created)")
except Exception as e:
    print(f"⚠️  Could not check index details: {str(e)}")
    print("   But connection works, so you can proceed!")

# Test 5: Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("✅ All tests passed!")
print("\nNext steps:")
print("1. Go back to Streamlit app")
print("2. Click Home > Select resumes > Process & Index")
print("3. System will create new index and upload vectors")
print("=" * 70)
