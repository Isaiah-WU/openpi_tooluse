import asyncio
import threading

from openpi.serving.websocket_policy_server import PolicyInferenceExecutor


def test_inference_executor_reuses_one_persistent_worker_thread():
    executor = PolicyInferenceExecutor()
    try:
        sync_thread = executor.run(threading.get_ident)
        async_thread = asyncio.run(executor.run_async(threading.get_ident))
    finally:
        executor.close()

    assert sync_thread == async_thread
    assert sync_thread != threading.get_ident()
