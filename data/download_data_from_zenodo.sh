#!/bin/bash

# Download and unzip 
curl https://zenodo.org/record/15474960/files/heavy_precessing_bbh_time_domain_data.zip --output "heavy_precessing_bbh_time_domain_data.zip"
unzip heavy_precessing_bbh_time_domain_data.zip

# Remove zip and Mac OSX files
rm heavy_precessing_bbh_time_domain_data.zip