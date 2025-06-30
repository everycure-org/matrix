#!/usr/bin/env python3
"""
Script demonstrating GCP project auto-detection functionality.

This script shows how the new auto-detection features work and can be used
to verify your gcloud configuration is set up correctly.
"""

import os

from matrix.utils.kubernetes import (
    can_talk_to_kubernetes,
    get_gcp_project_from_config,
    get_runtime_gcp_bucket,
    get_runtime_gcp_project_id,
    get_runtime_mlflow_url,
)


def main():
    print("🔍 GCP Auto-Detection Example")
    print("=" * 40)

    try:
        # 1. Get the active GCP project from gcloud config
        print("1. Auto-detecting GCP project from gcloud config...")
        project_id = get_gcp_project_from_config()
        print(f"   ✅ Active project: {project_id}")

        # 2. Get runtime project ID (with fallback to environment)
        print("\n2. Getting runtime project ID (with env fallback)...")
        runtime_project = get_runtime_gcp_project_id()
        print(f"   ✅ Runtime project: {runtime_project}")

        # 3. Auto-detect MLflow URL
        print("\n3. Auto-detecting MLflow URL...")
        mlflow_url = get_runtime_mlflow_url(runtime_project)
        print(f"   ✅ MLflow URL: {mlflow_url}")

        # 4. Auto-detect GCP bucket
        print("\n4. Auto-detecting GCP bucket...")
        bucket = get_runtime_gcp_bucket(runtime_project)
        print(f"   ✅ GCP bucket: {bucket}")

        # 5. Environment variables (if set)
        print("\n5. Environment variable overrides:")
        env_vars = ["RUNTIME_GCP_PROJECT_ID", "RUNTIME_GCP_BUCKET", "MLFLOW_URL"]

        for var in env_vars:
            value = os.environ.get(var)
            if value:
                print(f"   ⚙️  {var}: {value}")
            else:
                print(f"   ⭕ {var}: (not set - using auto-detection)")

        # 6. Test Kubernetes connectivity
        print("\n6. Testing Kubernetes connectivity...")
        try:
            can_connect = can_talk_to_kubernetes()
            if can_connect:
                print("   ✅ Kubernetes connection successful!")
            else:
                print("   ❌ Kubernetes connection failed")
        except Exception as e:
            print(f"   ❌ Kubernetes connection error: {e}")

        print("\n" + "=" * 40)
        print("✨ Auto-detection complete!")

        # Summary
        environment = "production" if "prod" in runtime_project else "development"
        print(f"📊 Summary:")
        print(f"   Environment: {environment}")
        print(f"   Project: {runtime_project}")
        print(f"   Bucket: {bucket}")
        print(f"   MLflow: {mlflow_url}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Make sure gcloud is installed: brew install google-cloud-sdk")
        print("   - Set your active project: gcloud config set project PROJECT_ID")
        print("   - Authenticate: gcloud auth login")


if __name__ == "__main__":
    main()
