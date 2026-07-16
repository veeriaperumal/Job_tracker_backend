import asyncio
import sys
from unittest.mock import AsyncMock, patch

# Add src and root to path
sys.path.append('.')
sys.path.append('./src')

from src.test.test_jobs import (
    test_generate_queries_without_profile,
    test_generate_queries_with_profile,
    test_normalize_results,
    test_remove_duplicates,
    test_api_jobs_search
)

def run_sync_test(test_func):
    try:
        test_func()
        print(f"PASS: {test_func.__name__}")
        return True
    except Exception as e:
        print(f"FAIL: {test_func.__name__}")
        import traceback
        traceback.print_exc()
        return False

async def run_async_test(test_func):
    try:
        await test_func()
        print(f"PASS: {test_func.__name__}")
        return True
    except Exception as e:
        print(f"FAIL: {test_func.__name__}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("Running Job Pipeline Unit Tests...")
    success = True
    
    # Sync tests
    success &= run_sync_test(test_generate_queries_without_profile)
    success &= run_sync_test(test_generate_queries_with_profile)
    success &= run_sync_test(test_normalize_results)
    success &= run_sync_test(test_remove_duplicates)
    
    # Async tests
    # Note: test_api_jobs_search is decorated with @patch, so we need to run it with patches.
    # The patch decorators are already on test_api_jobs_search, but since we are calling it directly
    # in an async loop, let's execute it directly.
    try:
        await test_api_jobs_search()
        print("PASS: test_api_jobs_search")
    except Exception as e:
        print("FAIL: test_api_jobs_search")
        import traceback
        traceback.print_exc()
        success = False
        
    if not success:
        print("Some tests failed!")
        sys.exit(1)
    else:
        print("All tests passed successfully!")

if __name__ == '__main__':
    asyncio.run(main())
