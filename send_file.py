# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) Copyright 2019 Enric Moreu. All Rights Reserved.

import requests

# Load multiple files
with open('demo_img.jpg', 'rb') as f:
    contents = f.read()
    
    files=dict(img1=contents)

    # Send files and parameters
    response = requests.post('http://localhost:5000/',
        files=files)

    # Save the received file
    with open('response_file.png', 'wb') as f:
        f.write(response.content)