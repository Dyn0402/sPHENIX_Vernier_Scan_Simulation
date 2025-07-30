#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 28 14:23 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/run_triggers

@author: Dylan Neff, dn277127
"""

import requests
from common_logistics import set_base_path


def main():
    """
    Pulls run trigger information from the sPHENIX run database. Writes to html file.
    Must be connected to campus network proxy to access the database --> ssh sph-tunnel to 3128 batch3.phy.bnl.gov:3128
    :return:
    """
    base_path = set_base_path() + 'Vernier_Scans/'
    # run_number = 69561
    run_number = 54733
    # run_number = 51195
    out_html_path = f'{base_path}run_{run_number}_trigger_details.html'

    # Construct the URL with the parameters
    base_url = "http://www.sphenix-intra.bnl.gov:7815/cgi-bin/trigger_details.py"
    proxies = {
        "http": "http://localhost:3128",
        "https": "http://localhost:3128"
    }

    params = {
        "run": run_number,
    }
    # Send a GET request to the URL
    response = requests.get(base_url, params=params, proxies=proxies)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        content = response.text
        print(content)
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved to {out_html_path}")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

    print('donzo')


if __name__ == '__main__':
    main()
