from django.urls import path ,include
from . import views
urlpatterns = [
    path('predict_reg/', views.predict_regression),
    path('predict_tree', views.predict_tree),
    path('predict_xgboost', views.predict_xgboost)

]