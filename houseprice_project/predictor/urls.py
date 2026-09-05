# predictor/urls.py
from django.urls import path
from .views import predict_price, prediction_status, health_check

urlpatterns = [
    path('predict/', predict_price, name='predict'),
    path('status/<uuid:task_id>/', prediction_status, name='prediction_status'),
    path('health/', health_check, name='health_check'),
]