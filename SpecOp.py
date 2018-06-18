import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('TkAgg') #absolutely essential!!!
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.table import Table
import sys                                                                           
from numpy import pi
from scipy import interpolate as interp
from scipy.optimize import curve_fit

try:
    import tkinter as tk
except ImportError:
    #possibly ModuleNotFoundError
    import Tkinter as tk

import time

def gui_process_select():
    args = ['Black and White', 'Grey']
    descriptions = ['Object is bright and mostly elliptical on a black background (Y/N?)',
                    'Object only visible in narrow wavelengths bands (Y/N?)']
    master = tk.Tk()
    master.wm_title('Select Extraction Process')
    for i in range(0, len(args)):
        tk.Label(master, text = args[i]).grid(row=i, column = 0)
        tk.Label(master, text = descriptions[i]).grid(row=i, column = 2)
        
    entries = []
    for i in range(0, len(args)):
        e_i = tk.Entry(master)
        entries.append(e_i)
        e_i.grid(row=i, column = 1)
        #e_i.insert(0, defaults[i])

    def read_args():
        vals = []
        for i in range(0, len(args)):
            vals.append(entries[i].get())

        global arg_list_NONLOCAL_Proc
        arg_list_NONLOCAL_Proc = vals
        master.quit()

    tk.Button(master, text = 'Submit', command = read_args).grid(row=len(args)+2, column=0)

    tk.mainloop()
    master.destroy()
    return arg_list_NONLOCAL_Proc

def gui_inputs_G():
    args = ['Method', 'P', 'Lambda min', 'Lambda max',  'reject', 'fout', 'fmt']
    defaults = ['auto', '1.0', '', '', 'False', '', 'ascii.basic']
    descriptions = ['Extraction algorithm, auto or manual',
                    'Power of S/N weighted average',
                    'Beginning of wavelength range (Angstroms, smallest possible if blank)',
                    'End of wavelength range (Angstroms, largest possible if blank)',
                    'Reject pixels with negative counts and |S/N| < 1/3',
                    'Name of output table (leave blank to not save a table)',
                    'Format of table (eg: ascii.basic, ascii.csv, fits, and more)']

    master = tk.Tk()
    master.wm_title('Grey Extraction Options')
    for i in range(0, len(args)):
        tk.Label(master, text = args[i]).grid(row=i, column = 0)
        tk.Label(master, text = descriptions[i]).grid(row=i, column = 2)
        
    entries = []
    for i in range(0, len(args)):
        e_i = tk.Entry(master)
        entries.append(e_i)
        e_i.grid(row=i, column = 1)
        e_i.insert(0, defaults[i])

    def read_args():
        vals = []
        for i in range(0, len(args)):
            vals.append(entries[i].get())

        global arg_list_NONLOCAL_G
        arg_list_NONLOCAL_G = vals
        master.quit()

    tk.Button(master, text = 'Submit', command = read_args).grid(row=len(args)+2, column=0)

    tk.mainloop()
    master.destroy()
    return arg_list_NONLOCAL_G

def gui_inputs_BW():

    args = ['Method', 'f', 'P', 'Lambda min', 'Lambda max', 'Lambda Intervals', 'N Contours', 'reject', 'fout', 'fmt']
    defaults = ['optimal', '0.95', '1.0', '', '', '1', '100', 'False', '', 'ascii.basic']
    descriptions = ['Extraction algorithm, simple or optimal',
                    'Fraction of light used to calculate include pixels',
                    'Power of S/N weighted average',
                    'Beginning of wavelength range (Angstroms, smallest possible if blank)',
                    'End of wavelength range (Angstroms, largest possible if blank)',
                    'Number of intervals into which wavelength range is divided',
                    'Number of contours used to find pixels satisfying enclosed light fraction.',
                    'Reject pixels with negative counts and |S/N| < 1/3',
                    'Name of output table (leave blank to not save a table)',
                    'Format of table (eg: ascii.basic, ascii.csv, fits, and more)']
        
    master = tk.Tk()
    master.wm_title('Black and White Extraction Options')
        
    for i in range(0, len(args)):
        tk.Label(master, text = args[i]).grid(row=i, column = 0)
        tk.Label(master, text = descriptions[i]).grid(row=i, column = 2)

    entries = []
    for i in range(0, len(args)):
        e_i = tk.Entry(master)
        entries.append(e_i)
        e_i.grid(row=i, column = 1)
        e_i.insert(0, defaults[i])

    def read_args():
        vals = []
        for i in range(0, len(args)):
            vals.append(entries[i].get())

        global arg_list_NONLOCAL_BW
        arg_list_NONLOCAL_BW = vals
        master.quit()
        
    #tk.Button(master, text = 'Run', command = master.quit).grid(row=7, column=1)
    tk.Button(master, text = 'Submit', command = read_args).grid(row=len(args)+2, column=0)

    tk.mainloop()
    master.destroy()
    return arg_list_NONLOCAL_BW


'''
class pix:
    def __init__(self, x, y, spec):
        self.x = x
        self.y = y
        self.img_x = x+1
        self.img_y = y+1
        self.coords = (self.x,self.y)
        self.img_coords = (self.img_x,self.img_y)
        self.include = False
        self.spec = spec

    def update_inc(self, inc_bool):
        self.include = inc_bool
'''
        
def get_data(fname):
    '''
    Load science data from a fits file and its error file
    '''
    
    errname = 'e.' + fname

    hdu = fits.open(fname)[0]
    errhdu = fits.open(errname)[0]
    
    global vert
    global horiz
    vert = hdu.header['NAXIS2'] #24
    horiz = hdu.header['NAXIS1'] #42

    lam_len = hdu.header['NAXIS3']
    lam_init = hdu.header['CRVAL3']
    lam_delt = hdu.header['CDELT3']
    lam_end = lam_init + lam_len*lam_delt

    sci_data = hdu.data
    err_data = errhdu.data

    lambdas = np.array([lam_init])
    for i in range(0, lam_len-1):
        lambdas = np.append(lambdas, lambdas[i]+lam_delt)

    return lam_len, lam_init, lam_delt, lam_end, sci_data, err_data, lambdas

def display_img(flat_img, title='image display'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(flat_img, cmap='gray')
    ax.set_xlabel('x pix')
    ax.set_ylabel('y pix')
    ax.set_title(title)
    return fig, ax

def centroid_aperture(included_pix, flat_img):
    #inclued_pix should be array of (y,x) tuples
    total_counts = 0.0
    ycom = 0.0
    xcom = 0.0

    for coord in included_pix:
        ycom += coord[0] * flat_img[coord]
        xcom += coord[1] * flat_img[coord]
        total_counts += flat_img[coord]
        
    ycom/=total_counts
    xcom/=total_counts

    return [xcom, ycom]

def centroid(flattened_image):
    '''
    Find "center of mass" centroid of a data cube
    '''
    
    flattened_image = np.sum(data[lam_min:lam_max,:,:], axis = 0)

    total_counts = np.sum(flattened_image)

    xcom = 0.0
    ycom = 0.0
    for j in range(0, vert):
        for i in range(0, horiz):
            xcom += i * flattened_image[j,i]
            ycom += j * flattened_image[j,i]

    xcom = xcom / total_counts
    ycom = ycom / total_counts
            
    return [xcom, ycom]

def manual_centroid(data, y_range, x_range, lam_min, lam_max):
    '''
    Find "center of mass" centroid of a data cube
    '''
    
    flattened_image = np.sum(data[lam_min:lam_max,:,:], axis = 0)

    total_counts = np.sum(flattened_image[y_range[0]:y_range[1],x_range[0]:x_range[1]])

    xcom = 0.0
    ycom = 0.0
    for j in range(y_range[0], y_range[1]):
        for i in range(x_range[0], x_range[1]):
            xcom += i * flattened_image[j,i]
            ycom += j * flattened_image[j,i]

    xcom = xcom / total_counts
    ycom = ycom / total_counts
            
    return [xcom, ycom]

'''
def find_rmax(data, xcom_float, ycom_float, f, horiz, vert):
 
    Find radius at which image contains f fraction of light
   

    xcom = int(np.round(xcom_float,0))
    ycom = int(np.round(ycom_float,0))

    #Simple sum along wavelength axis
    flattened_image = np.sum(data, axis = 0)

    xvals = np.arange(xcom, horiz)
    counts = flattened_image[ycom, xcom:]
    
    #Translate profile so it is positive everywhere
    baseline = np.abs(np.mean(flattened_image[ycom, horiz-3:horiz]))
    for i in range(0, len(counts)):
        counts[i] += baseline
    
    #Fit a spline to spatial profile 
    spline = interp.splrep(xvals, counts)

    xnew = np.linspace(xcom_float, horiz, 100)
    ynew = interp.splev(xnew, spline, der=0)
    
    #Integrate that spline to find r
    total_integral = interp.splint(xcom_float, horiz, spline)

    i = 0
    frac = 0
    while frac <= f:
        frac = interp.splint(xcom_float, xnew[i], spline) / total_integral
        i+=1
    i-=1
    rmax = xnew[i] - xcom_float
    
    ### plot in GUI ###
    master = tk.Tk()
    master.wm_title('Spatial Profile')
    fig = plt.figure(1)
    plt.plot(xvals, counts / total_integral, 'b')
    plt.plot(xnew, ynew, 'm')
    plt.plot((xnew[i], xnew[i]), (0, ynew[i]), c = 'r', linewidth = 2.0, linestyle = 'dashed')
    plt.plot((xcom_float, xcom_float), (0,ynew[0]), c = 'r', linewidth = 2.0, linestyle = 'dashed')
    plt.axhline(y=0, color = 'k', linestyle = 'dashed')
    plt.xlabel('Pixel X Value')
    plt.ylabel('Fractional Counts')
    plt.text(xnew[i] + (plt.xlim()[1]-plt.xlim()[0])*0.05,
             ynew[i] + (plt.ylim()[1]-plt.ylim()[0])*0.05,
             'radius = ' + str(round(rmax,3)) + ' pix')
    plt.text(plt.xlim()[0] + 0.75*(plt.xlim()[1]-plt.xlim()[0]),
             plt.ylim()[0] + 0.75*(plt.ylim()[1]-plt.ylim()[0]),
             'Centroid: \n' + '(' + str(round(xcom_float,3)) + ', ' + str(round(ycom_float,3)) + ')') 
    
    canvas = FigureCanvasTkAgg(fig, master=master)
    plot_widget = canvas.get_tk_widget()
    plot_widget.grid(row=0, column=0)
    tk.Button(master, text = 'Close', command = master.quit).grid(row=1, column = 0)
    master.mainloop()
    plt.close()
    master.destroy()
    
    return rmax
'''
def find_r_scatter(flattened_image, xcom, ycom):

    #flattened_image = np.sum(data, axis = 0)

    #print(xcom, ycom)

    r_vals = []
    c_vals = []

    for j in range(0, vert):
        for i in range(0, horiz):
            #removed +0.5 from i,j
            r_vals.append(distance_formula(i, xcom, j, ycom))
            c_vals.append(flattened_image[j,i])
    r_vals = np.insert(r_vals, 0, 0.)
    c_vals = np.insert(c_vals, 0, flattened_image[int(ycom),int(xcom)])

    #sort by ascending radial distance from centroid
    r_idxs = np.argsort(r_vals)
    r_vals = r_vals[r_idxs]
    c_vals = c_vals[r_idxs]
    
    #fit a Gaussian..
    def gauss(x, a, b, c):
        return a * np.exp((-(x-b)**2.)/(2*(c**2.)))

    #add baseline to make sky regions = 0 for fit
    r_percent = 0.10
    baseline = np.abs(np.mean(c_vals[int(len(c_vals)*(1.-r_percent)):]))
    #print('baseline:', baseline)
    c_vals+=baseline
    
    #curve fit, popt has optimal parameters
    spread_guess = distance_formula(horiz/2., horiz, vert/2., vert)/2.
    popt, pcov = curve_fit(gauss, r_vals, c_vals, p0=[max(c_vals),0.,spread_guess])
    #print(popt)

    print('Displaying spatial profile fit...')
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(r_vals, c_vals, s = 0.3, color = 'k')
    x_plot=np.linspace(0, max(r_vals), 50)
    ax.plot(x_plot, gauss(x_plot, *popt), color='r')
    ax.set_title('Spatial Profile Gaussian Fit')
    ax.set_xlabel('Radius (pix)')
    ax.set_ylabel('Counts')
    plt.show()
    blah = input('Press enter to proceed')
    plt.close()
    plt.ioff()

    #Now, the fit is really good, use FWHM as aperture?
    r_aperture = (2*np.log(2.))**0.5 * popt[2] #FWHM/2 definition
    
    return r_aperture

def distance_formula(x1, x2, y1, y2):
    '''
    distance between two points on cartesian plane
    '''
    dist = ( (x1-x2)**2 + (y1-y2)**2 )**(0.5)
    return dist

def calc_radius(contours, index, centroid):
    almost_vertices = contours.collections[index].get_paths()[0]
    points = almost_vertices.vertices
    x_conts = points[:,0]
    y_conts = points[:,1]

    r_vals = np.zeros(len(x_conts)-1)

    for i in range(0, len(r_vals)):
        r_vals[i] = distance_formula(centroid[0],x_conts[i],centroid[1],y_conts[i])

    r_avg = np.mean(r_vals)
    r_err = np.std(r_vals) / (float(len(r_vals))**0.5)

    return r_avg, r_err
        
def calc_deviation(counts, err, lambdas, lam_min, lam_max):
    lam_min_idx = np.argmin(np.abs(lambdas-lam_min))
    lam_max_idx = np.argmin(np.abs(lambdas-lam_max))

    sig = np.std(counts[lam_min_idx:lam_max_idx+1], ddof=1)
    mu = np.mean(counts[lam_min_idx:lam_max_idx+1])
    
    return mu, sig

def check_intersect(h_line, any_line):
    '''
    takes line segments defined by two points each, one must be horizontal
    checks if they intersect
    '''

    intersect = False
    
    if h_line[0,1] != h_line[1,1]:
        print("WOAH NOT HORIZONTAL")
        
    h_xsort = np.sort(h_line[:,0])
    h_yval = h_line[0,1]
    l_xsort = np.sort(any_line[:,0])
    l_ysort = np.sort(any_line[:,1])

    if l_ysort[0] <= h_yval and l_ysort[1] >= h_yval:
        if l_xsort[0] >= h_xsort[0]:
            intersect = True
        else:
            if any_line[0,0] > any_line[1,0]:
                temp = any_line[0]
                any_line[0] = any_line[1]
                any_line[1] = temp
            dy = any_line[1,1] - any_line[0,1]
            dx = any_line[1,0] - any_line[0,0]
            if dy == 0 or dx == 0:
                intersect = False
            else:
                m = dy/dx
                det = any_line[0,0] + (h_yval - any_line[0,1])/m
                if det >= h_xsort[0]:
                    intersect = True
                else:
                    intersect = False
    else:
        intersect = False

    return intersect

def check_contour(contours, i, bool_grid, pix_inside):
    almost_vertices = contours.collections[i].get_paths()[0]
    points = almost_vertices.vertices
    x_conts = points[:,0] + np.random.uniform(-0.001, 0.001) #bs, change
    y_conts = points[:,1] + np.random.uniform(-0.001, 0.001) #bs, change

    #pix_inside = []
    for x in range(0, horiz):
        for y in range(0, vert):

            if bool_grid[y,x] == 1:
                continue

            #check if line intersects with EVERY line segment
            #to do that, need to evaluate orientations

            #horiz_line = np.array([[x+0.5, y+0.5], [horiz+1., y+0.5]])
            horiz_line = np.array([[x+0.,y+0.],[horiz+0.5,y+0.]])
            n_intersections = 0

            for p in range(0, len(x_conts)-1):
                   
                poly_line = np.array([[x_conts[p], y_conts[p]], [x_conts[p+1], y_conts[p+1]]])
                
                if check_intersect(horiz_line, poly_line):
                    n_intersections+=1
                        
            if (n_intersections % 2) == 1:
                #the point is INSIDE the polygon, include it
                #print(x,y, 'INSIDE')
                #print
                pix_inside.append([y,x])
                bool_grid[y,x] = 1
            
    return bool_grid, pix_inside

def add_half(array):
    for i in range(0, array.shape[0]):
        for j in range(0, array.shape[1]):
            array[i,j] += 0.5
    return array
        
def contour_extraction(data, lam_min, lam_max, N_contours, f):
    '''
    resources:
    http://www.geeksforgeeks.org/orientation-3-ordered-points/
    http://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    http://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
    '''
    calc_rad = False
    stopwatch = False
    
    flattened_image = np.sum(data[lam_min:lam_max,:,:], axis = 0)
    total_counts = np.sum(flattened_image)

    g = np.mgrid[0:vert, 0:horiz]
    #y_coords = add_half(g[0]) #row coords
    #x_coords = add_half(g[1]) #column coords
    #y_coords = g[0]
    #x_coords = g[1]
    
    y_coords = np.arange(0,vert)
    x_coords =np.arange(0,horiz)

    if stopwatch:
        begin = time.time()
        
    n = N_contours

    plt.ion()
    fig, ax = display_img(flattened_image)
    contours = ax.contour(x_coords,y_coords,flattened_image,n)
    n = len(contours.collections)
    #print("n", n)
    
    enclosed_light = 0.0

    #if bool grid[y,x] = 1, the pixel has been included
    bool_grid = np.zeros([vert, horiz])
    pix_inside = []

    i_low = int(n-1)
    i_high = int(0)
    i = int(np.ceil((i_low+i_high)/2.0))

    loop = True
    
    while loop:
        #print("i", i)
        #re-cast as numpy array / list to ensure memoryless property 
        cached_bool_grid = np.array(bool_grid)
        cached_pix_inside = list(pix_inside)

        bool_grid, pix_inside = check_contour(contours, i, bool_grid, pix_inside)
        
        temp_sum = 0.0
        for point in pix_inside:
            temp_sum+=flattened_image[point[0],point[1]] 

        enclosed_light = float(temp_sum) / float(total_counts)
        #print("light", str(round(enclosed_light,3)))

        if enclosed_light < f:
            i_low = i
            i = int(np.ceil((i_low + i_high)/2.0))
            if np.abs(i_low-i_high)<=1:
                #print("BASE CASE")
                loop = False
                if calc_rad:
                    r_avg, r_err = calc_radius(contours, i, centroid(data, lam_min, lam_max))
                    print("r:", r_avg, "err:", r_err)
        else:
            i_high = i
            i = int(np.ceil((i_low+i_high)/2.0))
            bool_grid = np.array(cached_bool_grid)
            pix_inside = list(cached_pix_inside)

    if stopwatch:
        end = time.time()
        print('Time taken: ' + str(round(end-begin,3)))
        
    pix_inside = np.array(pix_inside)
    ax.plot(pix_inside[:,1], pix_inside[:,0], 'bo', markersize=2.) 
    ax.set_title('Enclosed Light Fraction: ' + str(round(enclosed_light,3)), fontsize=20)
    ax.set_xlabel('X pixel', fontsize=14)
    ax.set_ylabel('Y pixel', fontsize=14)
    ax.set_xlim( (0,horiz) )
    ax.set_ylim( (0,vert) ) 
    plt.show()
    proceed = check_YorN('Proceed with this process (Y/N)?')
    plt.close()
    plt.ioff()
    if not proceed:
        #abort this process entirely
        return None
    else:
        return pix_inside

def strict_distance_clip(xcom, ycom, r):
    
    #If pix center is within radius, include

    included_pix = []

    for i in range(0, horiz):
        for j in range(0, vert):
            x = i #removed +0.5
            y = j #removed +0.5
            distance = distance_formula(x, xcom, y, ycom)
            if distance <= r:
                included_pix.append( (j,i) )

    return included_pix


def loose_distance_clip(data, xcom, ycom, r):
    
    #If any part of pix is within radius, include
    

    included_pix = []

    for i in range(0, horiz):
        for j in range(0, vert):
            x = i #removed +0.5
            y = j #removed +0.5
            distance = distance_formula(x, xcom, y, ycom)
            if distance <= r:
                included_pix.append( (j,i) )
            else:
                pixdetect = False
                for a in np.linspace(-0.5, 0.5, 3):
                    for b in np.linspace(-0.5, 0.5, 3):
                        if distance_formula(x+a, xcom, y+b, ycom) <= r:
                            pixdetdect = True
                if pixdetect:
                    included_pix.append( (j,i) )

    return included_pix


        
def to_bool(astring):
    #why not just chech first letter for t/f or T/F?
    if astring[0] == 'T' or astring[0] == 't':
        return True
    elif astring[0] == 'F' or astring[0] == 'f':
        return False
    else:
        print('ERROR: invalid bool')

def check_YorN(YN_message):

    misunderstood = 'Input not understood. Type Y/y or N/n...'
    YN_bool= False #Yes:True, No:False
    understood = False

    while not understood:
        
        inputstr = input(YN_message)
        if len(inputstr)<1 or not inputstr.isalpha():
            pass
        elif inputstr[0]=='y' or inputstr[0]=='Y':
            YN_bool = True
            understood = True
        elif inputstr[0]=='n' or inputstr[0]=='N':
            YN_bool = False
            understood = True
        else:
            pass
        
        if not understood:
            print(misunderstood)

    print()
    return YN_bool

def check_number_input(msg, dtype, bounds=[]):

    #requires upper AND lower bound!!!
    error_dtype_msg = 'Try again. Numeric values only!'
    error_range_msg = 'Value out of bounds!'
    valid = False
    
    while not valid:
        num = input(msg)
        try:
            num = dtype(num)
        except ValueError:
            print(error_dtype_msg)
        else:
            if len(bounds)==2:
                if num<=bounds[-1] and num>=bounds[0]:
                    valid = True
                else:
                    print(error_range_msg)
                    print('Bounds: ', bounds)
            else:
                #Value unbounded, any number goes...
                valid = True
    print()
    return num
        

def weighted_sum_counts(included_pix, sci_data, err_data, lambdas, P, reject):

    total_weighted_counts = np.zeros(len(lambdas))
    total_counts = np.zeros(len(lambdas))
    total_err = np.zeros(len(lambdas), dtype = float)

    for i in range(0, len(lambdas)):

        #counts = 0
        numerator_sum = 0
        denom_sum = 0
        #err_sum = 0
        err_numer = 0
        err_denom = 0
        for coord_prime in included_pix:

            coord = tuple(coord_prime)

            if reject:
                if sci_data[i][coord] < 0 and np.abs(sci_data[i][coord] / err_data[i][coord]) < 1./3.:
                    continue 
                
            #counts += sci_data[i][coord]
            count = sci_data[i][coord]
            err = err_data[i][coord]
            SN = np.abs(count/err)
            numerator_sum += count * (SN**P)
            denom_sum += SN**P
            err_numer += (err**2) * ((SN**P)**2)
            err_denom += (SN**P)

        total_weighted_counts[i] = numerator_sum / denom_sum
        #total_counts[i] = counts
        total_err[i] = (err_numer / err_denom**2.)**(0.5)
        
    return lambdas, total_weighted_counts, total_err

def aperture_algorithm(sci_data, lambdas, lam_guess, y_guess, x_guess):
    upperbound = min([lam_guess-0, len(lambdas)-lam_guess])

    proceed = False
    while not proceed:
        wav_range = check_number_input('How many wavelength bins to flatten over? ', int,
                                       [2, 2*upperbound-1])
        wav_rad = int(wav_range/2)
    
        plt.ion()
        flattened_image = np.sum(sci_data[lam_guess-wav_rad:lam_guess+wav_rad,:,:], axis = 0)

        #find centroid of region
        '''
        idea: find max radius from center guess that doesn't go out of bounds,
        find all pix within that radius, do a centroid, call that the center
        '''
        max_r = min([horiz-x_guess, x_guess, vert-y_guess, y_guess])
        include_centroid = strict_distance_clip(x_guess, y_guess, max_r)
        xcom, ycom = centroid_aperture(include_centroid, flattened_image)

        #show centroid
        fig, ax = display_img(flattened_image,
                              'Centroid over '+str(wav_range)+' wavelength bins')
        rec = Rectangle((xcom-0.5,ycom-0.5),1,1,
                        linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rec)
        plt.show()
        print('Wavelength Range:',
              list(np.round([lambdas[lam_guess-wav_rad],lambdas[lam_guess+wav_rad]], 1)))
        print('Centroid (X,Y):', tuple(np.round((xcom, ycom),2)))
        proceed = check_YorN('Proceed (Y/N)? ')
        plt.close()
        print()
    plt.ioff()

    r_aperture = find_r_scatter(flattened_image, xcom, ycom,)
    included_pix = strict_distance_clip(xcom, ycom, r_aperture)

    x_pix = np.array(included_pix)[:,1]
    y_pix = np.array(included_pix)[:,0]

    #show included pix
    plt.ion()
    fig, ax = display_img(flattened_image, 'Pixels Within Aperture')
    ax.plot(x_pix, y_pix, 'bo', markersize=2.)
    circ = Circle((xcom, ycom), radius=r_aperture, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(circ)
    plt.show()
    blah = input('Press enter to proceed')
    plt.close()
    plt.ioff()
    
    return included_pix

def manual_extract(sci_data, err_data, lambdas, P, reject):

    #take their guess for x,y,lambda location
    proceed = False
    while not proceed:
        xcoord = check_number_input('X coordinate guess: ', float, [0,horiz-1])
        ycoord = check_number_input('Y coordinate guess: ', float, [0,vert-1])
        lamval = check_number_input('Wavelengths guess (Angstroms): ', float, [lambdas[0], lambdas[-1]])
        lamcoord = np.argmin(np.abs(lambdas-lamval))

        #display
        plt.ion()
        fig, ax = display_img(sci_data[lamcoord, :, :], 'Your Guess Coordinate')
        rec = Rectangle((xcoord-0.5,ycoord-0.5),1,1,
                         linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rec)
        proceed = check_YorN('Continue with current guess (Y/N)? ')
        plt.close()
        plt.ioff()

    #centroid over wavelength bins
    included_pix = aperture_algorithm(sci_data, lambdas, lamcoord, ycoord, xcoord)
    lambdas, total_weighted_counts, total_err = weighted_sum_counts(included_pix, sci_data, err_data,
                                                                    lambdas, P, reject)

    return lambdas, total_weighted_counts, total_err

def auto_extract(sci_data, err_data, lambdas, P, reject):

    #find object
    proceed = False
    stored_pix_vals = []
    stored_pix_idxs = []
    plt.ion()

    mask = np.zeros(sci_data.shape, dtype = int)
    ma_sci_data = ma.masked_array(sci_data, mask=mask)
    set_frame_params = True
    while not proceed:
        found = False
        if set_frame_params:
            wx = check_number_input('X background frame width: ', int, [1,int(horiz/2.)-1])
            wy = check_number_input('Y background frame width: ', int, [1,int(vert/2.)-1])
            sig_thresh = check_number_input('S/N threshold: ', float, [1,np.inf])
        while not found: #CORRECT THIS!!!
            max_idx = np.unravel_index(np.argmax(ma_sci_data), ma_sci_data.shape)
            #make sure these become user-defined
            #wx = 7 
            #wy = 4
            lamcoord, ycoord, xcoord = max_idx
            lamval = lambdas[max_idx[0]]
            if wx>xcoord or (horiz-wx)<=xcoord or wy>ycoord or (vert-wy)<=ycoord:
                #mask frame
                ma_sci_data.mask[lamcoord,:,:] = 1
                continue
            
            #estimate background
            bg_counts = np.array([])
            frame = sci_data[lamcoord,:,:] #masked version proves problematic
            bg_counts=np.append(bg_counts, frame[0:wy,:].flatten()) #top pannel
            bg_counts=np.append(bg_counts, frame[-wy:,:].flatten()) #bottome pannel
            bg_counts=np.append(bg_counts, frame[wy:-wy,0:wx].flatten()) #left pannel (exclduing corners)
            bg_counts=np.append(bg_counts, frame[wy:-wy,-wx:].flatten()) #right pannel (excluding corners)
            bg_mean = np.mean(bg_counts)
            bg_dev = np.std(bg_counts)
            
            #make sure pix and all neighbording pix have signal above N deviations
            increments = [0,-1,1]
            exit_loop = False
            for y in increments:
                if exit_loop:
                    break
                for x in increments:
                    if ma_sci_data[lamcoord,ycoord+y,xcoord+x] < (bg_mean+sig_thresh*bg_dev):
                        ma_sci_data.mask[lamcoord,:,:] = 1
                        exit_loop = True
                        found=False
                        break
                    else:
                        #object found, now display and ask if acceptable
                        found=True
        det_signif = (ma_sci_data[lamcoord,ycoord,xcoord]-bg_mean)/bg_dev

        #show image center guess
        fig, ax = display_img(frame, 'Current Coordinate Guess')
        rec1 = Rectangle((xcoord-0.5,ycoord-0.5),1,1,
                        linewidth=1,edgecolor='r',facecolor='none')
        rec2 = Rectangle((wx-0.5,wy-0.5),horiz-2*wx,vert-2*wy,
                        linewidth=1,edgecolor='w',facecolor='none')
        ax.add_patch(rec1)
        ax.add_patch(rec2)
        plt.show()

        #ask if it's good
        print('Image Coordinates (X,Y): '+'('+str(xcoord)+','+str(ycoord)+')')
        print('Wavelength (A): '+str(round(lamval,0)))
        print('Detection significance: ' +str(round(det_signif,0)))
        proceed_msg = 'Is the object contained in this frame (Y/N)? '
        proceed = check_YorN(proceed_msg)
        if proceed:
            center_guess_idx = max_idx
        else:
            #mask,update, keep going
            ma_sci_data.mask[lamcoord,:,:] = 1
            set_frame_params=check_YorN('Update background params (Y/N)? ')
            
        plt.close()
    plt.ioff()
    
    #centroid and find pixels within FWHM radius
    included_pix = aperture_algorithm(sci_data, lambdas, *center_guess_idx)
   
    #now, get the spectrum
    print('Performing weighted sum...')
    lambdas, total_weighted_counts, total_err = weighted_sum_counts(included_pix, sci_data, err_data,
                                                                    lambdas, P, reject)
    return lambdas, total_weighted_counts, total_err
            
        
def simple(sci_data, err_data, lambdas, N_contours, f, P, reject):

    print('Constructing spatial profile...', '\n') #NEED TO ALLOW FOR POSSIBILITY OF MULTIPLE INTERVALS
    included_pix = contour_extraction(sci_data, 0, len(lambdas), N_contours, f)
    
    #xcom, ycom = centroid(sci_data, vert, horiz, lam_min_idx, lam_max_idx)
    #rmax = find_r_scatter(sci_data, xcom, ycom, f, horiz, vert)

    '''
    if strict:
        included_pix = strict_distance_clip(sci_data, xcom, ycom, rmax)
    else:
        included_pix = loose_distance_clip(sci_data, xcom, ycom, rmax)
    '''
    
    print('Performing S/N weighted average...')

    lambdas, total_weighted_counts, total_err = weighted_sum_counts(included_pix, sci_data, err_data,
                                                                    lambdas, P, reject)
    '''
    total_weighted_counts = np.zeros(len(lambdas))
    total_counts = np.zeros(len(lambdas))
    total_err = np.zeros(len(lambdas), dtype = float)

    for i in range(lam_min_idx, lam_max_idx):

        #counts = 0
        numerator_sum = 0
        denom_sum = 0
        #err_sum = 0
        err_numer = 0
        err_denom = 0
        for coord_prime in included_pix:

            coord = tuple(coord_prime)

            if reject:
                if sci_data[i][coord] < 0 and np.abs(sci_data[i][coord] / err_data[i][coord]) < 1./3.:
                    continue 
                
            #counts += sci_data[i][coord]
            count = sci_data[i][coord]
            err = err_data[i][coord]
            SN = np.abs(count/err)
            numerator_sum += count * (SN**P)
            denom_sum += SN**P
            err_numer += (err**2) * ((SN**P)**2)
            err_denom += (SN**P)

        total_weighted_counts[i] = numerator_sum / denom_sum
        #total_counts[i] = counts
        total_err[i] = (err_numer / err_denom**2.)**(0.5)
    '''
    
    return lambdas, total_weighted_counts, total_err


def optimal(sci_data, err_data, lambdas, lam_intervals, N_contours, f, reject):

    #splits wavelength array into user define intervals 
    lambda_split = np.array_split(lambdas, lam_intervals)

    total_weighted_counts = np.array([], dtype= float)
    total_err = np.array([], dtype = float)

    for j in range(0, lam_intervals):
        
        print('Constructing spatial profile ' + str(j+1) + ' of ' + str(lam_intervals) + '...', '\n')
        lam_min = np.argmin(np.abs(lambdas-lambda_split[j][0]))
        lam_max = np.argmin(np.abs(lambdas-lambda_split[j][-1]))

        included_pix = contour_extraction(sci_data, lam_min, lam_max, N_contours, f)
        try:
            if included_pix == None:
                return lambdas, None, None
        except ValueError:
            pass
    
        #xcom, ycom = centroid(sci_data, vert, horiz, lam_min_idx, lam_max_idx)
        #rmax = find_r_scatter(sci_data, xcom, ycom, f, horiz, vert)

        '''
        if strict:
            included_pix = strict_distance_clip(sci_data, xcom, ycom, rmax)
        else:
            included_pix = loose_distance_clip(sci_data, xcom, ycom, rmax)
        '''
    
        print('Performing S/N weighted average...', '\n')

        interval_weighted_counts = np.zeros(len(lambda_split[j]), dtype = float)
        #interval_counts = np.zeros(len(lambdas), dtype= float)
        interval_err = np.zeros(len(lambda_split[j]), dtype = float)

        for i in range(lam_min, lam_max+1):

            
            numerator_sum = 0.0
            denom_sum = 0.0
            err_numer = 0.0
            var_denom = 0.0

            #Total counts in the contour-defined region
            region_counts = 0.0
            for coord_prime in included_pix:
                coord = tuple(coord_prime)
                region_counts+=sci_data[i][coord]
            
            for coord_prime in included_pix:

                coord = tuple(coord_prime)

                if reject:
                    if sci_data[i][coord] < 0 and np.abs(sci_data[i][coord] / err_data[i][coord]) < 1/3:
                        continue 
                
                count = sci_data[i][coord]
                err = err_data[i][coord]
                #Normalize the profile to the total counts in the region
                Profile = count/region_counts
                numerator_sum += (count * Profile) / err**2.0
                denom_sum += Profile**2.0 / err**2.0
                var_denom += Profile**2.0 / err**2.0

            interval_weighted_counts[i-lam_min] = numerator_sum / denom_sum
            interval_err[i-lam_min] = (1.0 / var_denom)**(0.5)
            
        total_weighted_counts = np.append(total_weighted_counts, interval_weighted_counts)
        total_err = np.append(total_err, interval_err)
    print(total_weighted_counts)
    print(total_err)
        
    return lambdas, total_weighted_counts, total_err


def extract(fname):
    '''
    Extract spectrum in high S/N region using S/N weighted average 
    '''

    #deviation = False
    print('Loading data...')
    lam_len, lam_init, lam_delt, lam_end, sci_data, err_data, lambdas = get_data(fname)

    #show user a flattened image to decide between BW or G
    plt.ion()
    fig, ax = display_img(np.sum(sci_data, axis=0), 'Image summed over all wavelengths')
    plt.show()
    blah = input('Press enter to proceed')
    plt.close()
    plt.ioff()
    
    print('Waiting for user inputs...', '\n')
    BlacknWhite, Grey = gui_process_select()
    if BlacknWhite=='':
        Grey = True
        BlacknWhite=False
    else:
        if BlacknWhite[0] == 'y' or BlacknWhite[0] == 'Y':
            Grey=False
            BlacknWhite=True
        elif Grey[0] == 'y' or Grey[0] == 'Y':
            Grey=True
            BlacknWhite=False
            
    plt.close()
    plt.ioff()
    
    if BlacknWhite:  
        method, f, P, lam_min, lam_max, lam_intervals, N_contours, reject, fout, fmt = gui_inputs_BW()

        f = float(f)
        P = float(P)
        lam_intervals = int(lam_intervals)
        N_contours = int(N_contours)
        reject = to_bool(reject)
        #write = to_bool(write)
        
    elif Grey:
        method, P, lam_min, lam_max, reject, fout, fmt = gui_inputs_G()

        P = float(P)
        reject = to_bool(reject)

    if len(fout) == 0:
        write = False
    else:
        write = True

    ### Clip Wavelength Range ###
    if lam_min == '':
        lam_min_idx = 0
    else:
        lam_min = float(lam_min)
        lam_min_idx = np.argmin(np.abs(lambdas-lam_min))

    if lam_max == '':
        lam_max_idx = len(lambdas)
    else:
        lam_max = float(lam_max)
        lam_max_idx = np.argmin(np.abs(lambdas-lam_max))

    lambdas = lambdas[lam_min_idx:lam_max_idx]
    sci_data = sci_data[lam_min_idx:lam_max_idx,:,:]

    #Call appropriate procedure
    if method == 'simple':
        lambdas, total_weighted_counts, total_err = simple(sci_data, err_data, lambdas,
                                                           N_contours, f, P, reject)
    if method == 'optimal':
        lambdas, total_weighted_counts, total_err = optimal(sci_data, err_data, lambdas,
                                                            lam_intervals, N_contours, f, reject)
        try:
            if total_weighted_counts == None:
                print('Switching to \'auto\' from the Grey process', '\n')
                method = 'auto'
        except ValueError:
            pass
    if method == 'auto':
        lambdas, total_weighted_counts, total_err = auto_extract(sci_data, err_data, lambdas,
                                                                 P, reject)
    if method == 'manual':
        lambdas, total_weighted_counts, total_err = manual_extract(sci_data, err_data, lambdas,
                                                                   P, reject)
    '''
    if deviation:
        mu, sig = calc_deviation(total_weighted_counts, total_err, lambdas, 6650, 6750)
        print('Deviation Test:', mu, sig)
        print('Error:', sig/mu)
    ''' 
        
    ### Plot Final Spectrum ###
    print('Displaying spectrum quicklook...', '\n')
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.ion()
    #plt.vlines(lambdas, [0], total_weighted_counts, colors = 'b')
    ax.errorbar(lambdas, total_weighted_counts, yerr = total_err, fmt = 'none', ecolor = 'k')
    ax.step(lambdas, total_weighted_counts, where='mid', color='b')
    ax.set_xlabel('Wavelength (Angstroms)')
    ax.set_ylabel('Weighted Counts')
    ax.set_title('Spectrum Quick Look')
    
    '''
    canvas = FigureCanvasTkAgg(fig, master=master)
    plot_widget = canvas.get_tk_widget()
    plot_widget.grid(row=0, column=0)
    tk.Button(master, text = 'Close', command = master.quit).grid(row=1, column = 0)
    master.mainloop()
    '''
    plt.show()
    input('Press enter to save/return spectrum')
    plt.close()
    plt.ioff()
    #plt.close()
    #master.destroy()
    
    if write:
        print('Writing data to ' + fmt + ' table with name ' + fout + '...', '\n')
        if fmt == 'fits':
            #call fits writing function
            write_fits_table(fname, fout, lambdas, total_weighted_counts, total_err)
            return
        elif len(fmt)>5:
            if fmt[0:5] == 'ascii':
                #call header writing function
                tout = Table(data = [lambdas, total_weighted_counts, total_err], names = ['Wavelength', 'Counts', 'Error'])
                tout.write(fout, format = fmt)
                write_ascii_header(fname, fout)
        else:
            tout = Table(data = [lambdas, total_weighted_counts, total_err], names = ['Wavelength', 'Counts', 'Error'])
            tout.write(fout, format = fmt)
            return
    
    else:
        print('Returning arrays of wavelength, weighted counts, and errors...', '\n')
        return lambdas, total_weighted_counts, total_err

def write_fits_table(fname, fout, lambdas, total_weighted_counts, total_err):
    #modified from http://docs.astropy.org/en/stable/io/fits/
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='Wavelength', array=lambdas, format ='E'),
                                         fits.Column(name='Counts', array=total_weighted_counts, format='E'),
                                         fits.Column(name='Error', array=total_err, format='E')])
    hdr = fits.getheader(fname)
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(fout)
    return

def write_ascii_header(fname, fout):
    keywords = ['DATE', 'TELRA', 'TELDEC', 'ZD', 'EXPTIME']
    kw_len = 8
    eql = '= '
    val_len = 20
    cmt_len = 49 #give space for '#' at beginning of line (otherwise 50)
    hdr = fits.getheader(fname)
    vals = []
    comments = []
    for kw in keywords:
        vals.append(str(hdr[kw]))
        comments.append(' / ' + hdr.comments[kw])
    for i in range(len(keywords)):
        while len(keywords[i]) < kw_len:
            keywords[i]+=' '
        while len(vals[i]) < val_len:
            vals[i]+=' '
        while len(comments[i]) < cmt_len:
            comments[i]+=' '

    header_string = ''
    for i in range(len(keywords)):
        header_string+='#'+keywords[i]
        header_string+=eql
        header_string+=vals[i]
        header_string+=comments[i]
    header_string+='\n'

    #now, header_string is correct, 80 character lines..
    #figure out how to write into a file correctly
    
    f = open(fout, "r")
    contents = f.readlines()
    f.close()
    
    contents[0] = '#' + contents[0] #add octothorpe before column names
    contents.insert(0, header_string) #put header at very top of file

    f = open(fout, "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()
    
    return
        
    

############### MAIN ###############
#calls extract through command line#
####################################
if len(sys.argv) == 2:
    cmnd_fname = sys.argv[1]
    extract(cmnd_fname)
else:
    print('Error, script takes one argument, the filename.')
    print('example:')
    print('$ python SpecOp.py Cu*.fits')


    

