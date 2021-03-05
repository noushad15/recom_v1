from django.shortcuts import render
from django.http import  Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import sys
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Create your views here.
# data = pd.read_csv("l2.csv")

@api_view(["POST", "GET"])
def IdealWeight(data_get):
    global data

    try:
        # print(data_get.body.decode())
        data_get1 = data_get.body.decode()
        # print(type(data_get1))

        out = subprocess.run([sys.executable, "Phase1_recommendation1.py", data_get1], shell=False, capture_output=True, text=True)
        # print(out)
        return JsonResponse(out.stdout,safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)