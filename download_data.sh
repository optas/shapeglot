#!/bin/bash
# Download the data collected & used in ShapeGlot (~218MB)
# We assume you have already accepted the Terms Of Use, else please visit: https://forms.gle/2cd4U9zdBH7r9PyTA

#DATA_LINK=<REPLACE WITH LINK PROVIDED after you accepted the terms of use of ShapeGlot>
wget $DATA_LINK
unzip data.zip
rm data.zip