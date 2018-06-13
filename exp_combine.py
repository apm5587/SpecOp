import numpy as np
from astropy.io import fits

def add(exp_names):
    
    #exp_names should be in chronological order
    exp_names = np.array(exp_names)
    #get names of error files
    e_exp_names = np.array([])
    for fname in exp_names:
        e_exp_names = np.append(e_exp_names, 'e.'+fname)

    #np.empty(len(exp_names))
    data = []
    headers = []
    e_data = []
    e_headers = []

    for i in range(0, len(exp_names)):
        data.append(fits.getdata(exp_names[i]))
        e_data.append(fits.getdata(e_exp_names[i]))
        headers.append(fits.getheader(exp_names[i]))
        e_headers.append(fits.getheader(e_exp_names[i]))

    #add data
    combined_data = np.zeros(data[0].shape)
    e_combined_data = np.zeros(e_data[0].shape)
    for exp in data:
        combined_data+=exp
    for exp in e_data:
        e_combined_data+=exp #do I need to do this in quadrature?

    #update header exposure times
    keywords = ['REXPTIME', 'PEXPTIME', 'DARKTIME', 'READTIME', 'EXPTIME']

    base_header = headers[0]
    e_base_header = e_headers[0]
    
    base_vals = np.zeros(len(keywords), dtype=float)
    e_base_vals = np.zeros(len(keywords), dtype=float)

    for i in range(0, len(keywords)):
        base_vals[i] = base_header[keywords[i]]
        e_base_vals[i] = e_base_header[keywords[i]]
        
    for i in range(0, len(keywords)):
        for j in range(1, len(exp_names)):
            base_vals[i] += headers[j][keywords[i]]
            e_base_vals[i] += e_headers[j][keywords[i]]

    #Now, update base_header with new base_vals
    for i in range(0, len(keywords)):
        base_header[keywords[i]] = base_vals[i]
        e_base_header[keywords[i]] = e_base_vals[i]

    #save as fits
    new_name = 'full_exp_' + exp_names[0]
    new_e_name = 'e.full_exp_' + exp_names[0]
    fits.writeto(new_name, combined_data, base_header)
    fits.writeto(new_e_name, e_combined_data, e_base_header)

    return

    
