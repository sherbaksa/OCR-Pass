#!/usr/bin/env python3
import os
import sys
import subprocess

def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 11:
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.11 or higher required")
        return False

def check_docker():
    """Проверка наличия Docker"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"Docker: {result.stdout.strip()}")
        
        # Проверка Docker Compose
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"Docker Compose: {result.stdout.strip()}")
        print("✅ Docker and Docker Compose OK")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker or Docker Compose not found or not working")
        return False

def check_env_file():
    """Проверка файла .env"""
    if os.path.exists('.env'):
        print("✅ .env file exists")
        # Проверка критических переменных
        with open('.env', 'r') as f:
            content = f.read()
            required_vars = ['MINIO_ACCESS_KEY', 'MINIO_SECRET_KEY', 'MINIO_BUCKET_NAME']
            for var in required_vars:
                if f"{var}=" in content:
                    print(f"  ✅ {var} is set")
                else:
                    print(f"  ⚠️  {var} is not set in .env")
        return True
    elif os.path.exists('.env.example'):
        print("⚠️  .env file not found, but .env.example exists")
        print("   Please copy .env.example to .env and fill in the values")
        return False
    else:
        print("❌ No .env or .env.example file found")
        return False

def check_project_structure():
    """Проверка структуры проекта"""
    required_dirs = ['backend', 'infra', 'scripts']
    required_files = [
        'backend/main.py',
        'backend/requirements.txt',
        'backend/Dockerfile',
        'docker-compose.yml',
        'infra/postgres/init.sql'
    ]
    
    print("Checking project structure:")
    
    all_ok = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ Directory {dir_path}/ exists")
        else:
            print(f"❌ Directory {dir_path}/ missing")
            all_ok = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} missing")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 50)
    print("Environment Check for Passport OCR Service")
    print("=" * 50)
    
    checks = [
        ("Python version", check_python_version),
        ("Docker", check_docker),
        ("Environment file", check_env_file),
        ("Project structure", check_project_structure)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}:")
        result = check_func()
        results.append((check_name, result))
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! You can proceed with:")
        print("   docker-compose up --build")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
