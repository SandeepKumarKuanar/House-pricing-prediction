import os
import json
import uuid
import threading
import logging

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'house_price_model.joblib')
COLUMNS_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'training_columns.json')

# Global model state with thread-safe locking
_model = None
_training_columns = None
_model_lock = threading.Lock()
_model_loading = False
_model_loaded_event = threading.Event()
_load_error = None

# In-memory task store (fine for single-worker; swap for Redis if scaling)
_tasks = {}
_tasks_lock = threading.Lock()


def _load_model():
    """Load model and training columns into global variables."""
    global _model, _training_columns, _model_loading, _load_error
    try:
        _load_error = None
        logger.info("Loading ML model...")
        loaded_model = joblib.load(MODEL_PATH)
        with open(COLUMNS_PATH, 'r') as f:
            loaded_columns = json.load(f)
        with _model_lock:
            _model = loaded_model
            _training_columns = loaded_columns
        logger.info("ML model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load ML model: %s", e)
        _load_error = str(e)
    finally:
        _model_loading = False
        _model_loaded_event.set()


def _ensure_model_loaded():
    """Trigger background model load if not already loaded or loading."""
    global _model_loading
    with _model_lock:
        if _model is not None:
            return
        if not _model_loading:
            _model_loading = True
            _model_loaded_event.clear()
            threading.Thread(target=_load_model, daemon=True).start()


def _wait_for_model(timeout=120):
    """Block until the model is loaded. Returns True if ready, else False."""
    _model_loaded_event.wait(timeout=timeout)
    with _model_lock:
        if _model is not None:
            return True
        if _load_error:
            raise RuntimeError(_load_error)
    return False


def _get_model_status():
    """Return (is_ready, is_loading)."""
    with _model_lock:
        return _model is not None, _model_loading


def _run_prediction(task_id, data):
    """Wait for the model, run prediction in background, store result for polling."""
    try:
        if not _wait_for_model():
            with _tasks_lock:
                _tasks[task_id] = {'status': 'error', 'error': 'Model load timed out.'}
            return

        with _model_lock:
            model = _model
            training_columns = _training_columns

        predicted_price = _do_predict(model, training_columns, data)

        with _tasks_lock:
            _tasks[task_id] = {'status': 'ready', 'predicted_price': predicted_price}
    except Exception as e:
        with _tasks_lock:
            _tasks[task_id] = {'status': 'error', 'error': str(e)}


def _do_predict(model, training_columns, data):
    """Apply the same pre-processing as training, then make a prediction."""
    input_df = pd.DataFrame([data])

    cols_to_log = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_basement', 'yr_built']
    for col in cols_to_log:
        input_df[col] = np.log(input_df[col].astype(float) + 1)

    input_df['living_quality'] = input_df['view'] + (input_df['grade'] / 2) + input_df['condition']
    input_df = input_df.drop(columns=['view', 'condition', 'grade'])
    input_df = input_df[training_columns]

    prediction = model.predict(input_df)
    return float(prediction[0])


@api_view(['POST'])
def predict_price(request):
    try:
        data = request.data
        is_ready, _ = _get_model_status()

        if is_ready:
            # Model already in memory: return prediction immediately
            with _model_lock:
                model = _model
                training_columns = _training_columns
            predicted_price = _do_predict(model, training_columns, data)
            return Response({'predicted_price': predicted_price})

        # Model not loaded: kick off background load/launch and return 202
        _ensure_model_loaded()
        task_id = str(uuid.uuid4())

        with _tasks_lock:
            _tasks[task_id] = {'status': 'loading'}

        threading.Thread(target=_run_prediction, args=(task_id, data), daemon=True).start()

        return Response(
            {'task_id': task_id, 'message': 'Model is warming up. Poll /api/status/<task_id>/'},
            status=202,
        )

    except Exception as e:
        return Response({'error': str(e)}, status=400)


@api_view(['GET'])
def prediction_status(request, task_id):
    """Poll endpoint: 202 while loading, 200 when ready, 404 if unknown."""
    task_id = str(task_id)  # URL converter yields uuid.UUID; store keys as str
    with _tasks_lock:
        task = _tasks.get(task_id)

    if task is None:
        return Response({'error': 'Unknown task_id'}, status=404)

    if task['status'] == 'loading':
        return Response({'status': 'loading', 'message': 'Model is warming up. Please retry.'}, status=202)

    # Clean up completed/errored tasks before responding
    with _tasks_lock:
        _tasks.pop(task_id, None)

    if task['status'] == 'error':
        return Response({'error': task['error']}, status=500)

    # status == 'ready'
    return Response({'predicted_price': task['predicted_price']})


def health_check(request):
    """Lightweight health check for keep-alive pings."""
    is_ready, is_loading = _get_model_status()
    return JsonResponse({
        'status': 'ok',
        'model_loaded': is_ready,
        'model_loading': is_loading,
    })